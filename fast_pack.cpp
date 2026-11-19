// Fast multi-threaded JSONL -> HalfKP-v3 .npy shard builder for HARENN
// Compile: g++ -O3 -std=c++17 -pthread fast_pack.cpp -o fast_pack
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static constexpr int HKV3_PLANES_PER_BUCKET = 11 * 64; // 704
static constexpr int HKV3_NUM_FEATURES = 32 * HKV3_PLANES_PER_BUCKET; // 22528
static constexpr int MAX_ACTIVE_FEATURES = 32;
static constexpr uint16_t PADDING_FEATURE_IDX = HKV3_NUM_FEATURES;
static constexpr double WDL_SCALE_PAWNS = 2.60;

struct Record {
    uint64_t hash;
    uint16_t us[32];
    uint16_t them[32];
    float eval_pawns;
    float wdl;
    uint8_t eval_mask;
    float tau;
    float rho;
    float rs;
    uint16_t bm_lbl;
    uint16_t sm_lbl;
};

static inline uint64_t hash_bytes_64(const char* data, size_t len) {
    // 64-bit FNV-1a + Murmur3 finalizer (fast, zero-alloc, ~0 collision at 30M)
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint8_t>(data[i]);
        h *= 1099511628211ULL;
    }
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

static inline int p2ch_white(char ch) {
    switch (ch) {
        case 'P': return 0; case 'N': return 1; case 'B': return 2; case 'R': return 3; case 'Q': return 4;
        case 'p': return 5; case 'n': return 6; case 'b': return 7; case 'r': return 8; case 'q': return 9; case 'k': return 10;
        default: return -1;
    }
}

static inline int p2ch_black(char ch) {
    switch (ch) {
        case 'p': return 0; case 'n': return 1; case 'b': return 2; case 'r': return 3; case 'q': return 4;
        case 'P': return 5; case 'N': return 6; case 'B': return 7; case 'R': return 8; case 'Q': return 9; case 'K': return 10;
        default: return -1;
    }
}

static bool is_sq_attacked(const char grid[64], int sq, bool by_white) {
    int r = sq >> 3, f = sq & 7;
    if (by_white) {
        if (r > 0) {
            if (f > 0 && grid[((r - 1) << 3) + f - 1] == 'P') return true;
            if (f < 7 && grid[((r - 1) << 3) + f + 1] == 'P') return true;
        }
    } else {
        if (r < 7) {
            if (f > 0 && grid[((r + 1) << 3) + f - 1] == 'p') return true;
            if (f < 7 && grid[((r + 1) << 3) + f + 1] == 'p') return true;
        }
    }
    static constexpr int kn_dr[8] = {-2,-2,-1,-1,1,1,2,2};
    static constexpr int kn_df[8] = {-1,1,-2,2,-2,2,-1,1};
    char kn = by_white ? 'N' : 'n';
    for (int i = 0; i < 8; ++i) {
        int nr = r + kn_dr[i], nf = f + kn_df[i];
        if ((unsigned)nr < 8u && (unsigned)nf < 8u && grid[(nr << 3) + nf] == kn) return true;
    }
    static constexpr int kg_dr[8] = {-1,-1,-1,0,0,1,1,1};
    static constexpr int kg_df[8] = {-1,0,1,-1,1,-1,0,1};
    char kg = by_white ? 'K' : 'k';
    for (int i = 0; i < 8; ++i) {
        int nr = r + kg_dr[i], nf = f + kg_df[i];
        if ((unsigned)nr < 8u && (unsigned)nf < 8u && grid[(nr << 3) + nf] == kg) return true;
    }
    static constexpr int or_dr[4] = {-1,1,0,0};
    static constexpr int or_df[4] = {0,0,-1,1};
    char rk = by_white ? 'R' : 'r', qn = by_white ? 'Q' : 'q', bs = by_white ? 'B' : 'b';
    for (int i = 0; i < 4; ++i) {
        int nr = r + or_dr[i], nf = f + or_df[i];
        while ((unsigned)nr < 8u && (unsigned)nf < 8u) {
            char p = grid[(nr << 3) + nf];
            if (p != 0) { if (p == rk || p == qn) return true; break; }
            nr += or_dr[i]; nf += or_df[i];
        }
    }
    static constexpr int dg_dr[4] = {-1,-1,1,1};
    static constexpr int dg_df[4] = {-1,1,-1,1};
    for (int i = 0; i < 4; ++i) {
        int nr = r + dg_dr[i], nf = f + dg_df[i];
        while ((unsigned)nr < 8u && (unsigned)nf < 8u) {
            char p = grid[(nr << 3) + nf];
            if (p != 0) { if (p == bs || p == qn) return true; break; }
            nr += dg_dr[i]; nf += dg_df[i];
        }
    }
    return false;
}

static bool infer_is_quiet(std::string_view board_fen, char turn_char, std::string_view bm_uci, double ev_cp_raw, double rho) {
    if (std::abs(ev_cp_raw) >= 1000.0 || rho >= 0.75) return false;
    char grid[64] = {};
    int wk = -1, bk = -1, r = 7, f = 0;
    for (char ch : board_fen) {
        if (ch == '/') { r--; f = 0; }
        else if (ch >= '1' && ch <= '8') { f += (ch - '0'); }
        else {
            if (r < 0 || r >= 8 || f < 0 || f >= 8) return false;
            int sq = (r << 3) + f;
            grid[sq] = ch;
            if (ch == 'K') wk = sq;
            else if (ch == 'k') bk = sq;
            f++;
        }
    }
    if (wk < 0 || bk < 0) return false;
    bool us_white = (turn_char == 'w' || turn_char == 'W');
    int our_king = us_white ? wk : bk;
    int opp_king = us_white ? bk : wk;
    if (is_sq_attacked(grid, our_king, !us_white)) return false;

    if (bm_uci.size() >= 4) {
        if (bm_uci.size() >= 5) return false;
        int f1 = bm_uci[0] - 'a', r1 = bm_uci[1] - '1';
        int f2 = bm_uci[2] - 'a', r2 = bm_uci[3] - '1';
        if ((unsigned)f1 < 8u && (unsigned)r1 < 8u && (unsigned)f2 < 8u && (unsigned)r2 < 8u) {
            int sq1 = (r1 << 3) + f1, sq2 = (r2 << 3) + f2;
            char moving = grid[sq1], target = grid[sq2];
            if (target != 0) {
                if (!((moving == 'K' && target == 'R') || (moving == 'k' && target == 'r'))) return false;
            }
            if ((moving == 'P' || moving == 'p') && f1 != f2) return false;
            grid[sq1] = 0;
            grid[sq2] = moving;
            if (is_sq_attacked(grid, opp_king, us_white)) return false;
        }
    }
    return true;
}

static inline void skip_ws(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
}

static bool parse_json_line(const char* line, size_t len, Record& out) {
    const char* p = line;
    const char* end = line + len;
    skip_ws(p, end);
    if (p >= end || *p != '{') return false;
    ++p;

    std::string_view fen, bm_uci;
    int stm_override = -1;
    bool has_eval = false; double ev_raw = 0.0;
    bool has_wdl = false; double wdl_raw = 0.0;
    bool has_gr = false; double gr_raw = 0.0;
    bool has_quiet = false; bool quiet_val = false;
    double tau = 0.5, rho = 0.3, rs = 0.5;
    int bm_lbl = 0, sm_lbl = 0;

    auto read_str = [&](std::string_view& sv) -> bool {
        if (p >= end || *p != '"') return false;
        ++p;
        const char* s = p;
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) p += 2;
            else ++p;
        }
        if (p >= end) return false;
        sv = std::string_view(s, static_cast<size_t>(p - s));
        ++p;
        return true;
    };

    auto skip_val = [&]() {
        skip_ws(p, end);
        if (p >= end) return;
        if (*p == '"') { std::string_view tmp; read_str(tmp); return; }
        if (*p == '{' || *p == '[') {
            int depth = 0; bool in_s = false;
            while (p < end) {
                char c = *p++;
                if (in_s) { if (c == '\\' && p < end) ++p; else if (c == '"') in_s = false; }
                else { if (c == '"') in_s = true; else if (c == '{' || c == '[') ++depth; else if (c == '}' || c == ']') { if (--depth == 0) break; } }
            }
            return;
        }
        while (p < end && *p != ',' && *p != '}') ++p;
    };

    auto read_num = [&](double& v) -> bool {
        skip_ws(p, end);
        if (p >= end) return false;
        if (*p == '"') {
            std::string_view sv;
            if (!read_str(sv)) return false;
            char buf[64];
            size_t n = std::min(sv.size(), sizeof(buf) - 1);
            std::memcpy(buf, sv.data(), n); buf[n] = 0;
            char* ep = nullptr;
            v = std::strtod(buf, &ep);
            return ep != buf;
        }
        char* ep = nullptr;
        v = std::strtod(p, &ep);
        if (ep == p) { skip_val(); return false; }
        p = ep;
        return true;
    };

    while (p < end) {
        skip_ws(p, end);
        if (p >= end || *p == '}') break;
        if (*p == ',') { ++p; continue; }
        std::string_view key;
        if (!read_str(key)) return false;
        skip_ws(p, end);
        if (p >= end || *p != ':') return false;
        ++p;
        skip_ws(p, end);
        if (p >= end) break;

        if (key == "fen" || key == "FEN" || key == "fen_string" || key == "board") {
            if (*p == '"') read_str(fen); else skip_val();
        } else if (key == "stm" || key == "turn" || key == "side_to_move" || key == "to_move") {
            if (*p == '"') {
                std::string_view sv; read_str(sv);
                if (sv == "b" || sv == "black" || sv == "1" || sv == "true") stm_override = 1;
                else if (sv == "w" || sv == "white" || sv == "0" || sv == "false") stm_override = 0;
            } else {
                double v = 0;
                if (read_num(v)) stm_override = (static_cast<int>(v) != 0) ? 1 : 0;
            }
        } else if (key == "eval_score" || key == "eval" || key == "score" || key == "cp" || key == "evaluation") {
            double v = 0;
            if (read_num(v)) { ev_raw = v; has_eval = true; }
        } else if (key == "tau") {
            double v = 0.5; if (read_num(v)) tau = std::clamp(v, 0.0, 1.0);
        } else if (key == "rho") {
            double v = 0.3; if (read_num(v)) rho = std::clamp(v, 0.0, 1.0);
        } else if (key == "rs") {
            double v = 0.5; if (read_num(v)) rs = std::clamp(v, 0.0, 1.0);
        } else if (key == "best_move_label") {
            double v = 0; if (read_num(v)) bm_lbl = std::clamp(static_cast<int>(v), 0, 767);
        } else if (key == "sec_move_label") {
            double v = 0; if (read_num(v)) sm_lbl = std::clamp(static_cast<int>(v), 0, 767);
        } else if (key == "best_move") {
            if (*p == '"') read_str(bm_uci); else skip_val();
        } else if (key == "is_quiet") {
            if (*p == 't') { quiet_val = true; has_quiet = true; p += 4; }
            else if (*p == 'f') { quiet_val = false; has_quiet = true; p += 5; }
            else if (*p == '"') {
                std::string_view sv; read_str(sv);
                quiet_val = (sv == "true" || sv == "1" || sv == "yes" || sv == "True");
                has_quiet = true;
            } else if (*p == 'n') { skip_val(); }
            else { double v = 0; if (read_num(v)) { quiet_val = (v != 0.0); has_quiet = true; } }
        } else if (key == "wdl") {
            if (*p == 'n') skip_val();
            else { double v = 0; if (read_num(v)) { wdl_raw = std::clamp(v, 0.0, 1.0); has_wdl = true; } }
        } else if (key == "game_result") {
            if (*p == 'n') skip_val();
            else { double v = 0; if (read_num(v)) { gr_raw = std::clamp(v / 2.0, 0.0, 1.0); has_gr = true; } }
        } else {
            skip_val();
        }
    }

    if (fen.empty() || !has_eval) return false;

    // Split fen into parts[0..3]
    int n_parts = 0;
    size_t pos_i = 0;
    std::string_view parts[4];
    while (pos_i < fen.size() && n_parts < 4) {
        while (pos_i < fen.size() && fen[pos_i] == ' ') ++pos_i;
        if (pos_i >= fen.size()) break;
        size_t s = pos_i;
        while (pos_i < fen.size() && fen[pos_i] != ' ') ++pos_i;
        parts[n_parts++] = fen.substr(s, pos_i - s);
    }
    if (n_parts < 2) return false;

    std::string_view board_key = (n_parts >= 4)
        ? fen.substr(0, static_cast<size_t>((parts[3].data() + parts[3].size()) - fen.data()))
        : fen;
    out.hash = hash_bytes_64(board_key.data(), board_key.size());

    int stm = (stm_override >= 0) ? stm_override : ((parts[1] == "b") ? 1 : 0);

    // Parse board FEN into HalfKP-v3
    struct PItem { uint8_t sq; char ch; };
    PItem pcs[32];
    int npcs = 0, w_ksq = -1, b_ksq = -1;
    int r = 7, f = 0, slashes = 0;
    for (char ch : parts[0]) {
        if (ch == '/') { r--; f = 0; slashes++; }
        else if (ch >= '1' && ch <= '8') { f += (ch - '0'); }
        else {
            if ((unsigned)r >= 8u || (unsigned)f >= 8u) return false;
            int sq = (r << 3) + f;
            if (ch == 'K') w_ksq = sq;
            else if (ch == 'k') b_ksq = sq;
            if (npcs < 32) pcs[npcs++] = {static_cast<uint8_t>(sq), ch};
            f++;
        }
    }
    if (slashes != 7 || w_ksq < 0 || b_ksq < 0) return false;

    int w_mirror = ((w_ksq & 7) >= 4) ? 7 : 0;
    int w_knorm = w_ksq ^ w_mirror;
    int w_kb = ((w_knorm >> 3) * 4 + (w_knorm & 3)) * HKV3_PLANES_PER_BUCKET;

    int b_korient = b_ksq ^ 56;
    int b_mirror = ((b_korient & 7) >= 4) ? 7 : 0;
    int b_knorm = b_korient ^ b_mirror;
    int b_kb = ((b_knorm >> 3) * 4 + (b_knorm & 3)) * HKV3_PLANES_PER_BUCKET;

    uint16_t w_arr[32], b_arr[32];
    for (int i = 0; i < 32; ++i) { w_arr[i] = PADDING_FEATURE_IDX; b_arr[i] = PADDING_FEATURE_IDX; }
    int w_cnt = 0, b_cnt = 0;
    for (int i = 0; i < npcs; ++i) {
        uint8_t sq = pcs[i].sq;
        char ch = pcs[i].ch;
        if (ch != 'K' && w_cnt < 32) {
            int cw = p2ch_white(ch);
            if (cw >= 0) w_arr[w_cnt++] = static_cast<uint16_t>(w_kb + cw * 64 + (sq ^ w_mirror));
        }
        if (ch != 'k' && b_cnt < 32) {
            int cb = p2ch_black(ch);
            if (cb >= 0) b_arr[b_cnt++] = static_cast<uint16_t>(b_kb + cb * 64 + ((sq ^ 56) ^ b_mirror));
        }
    }
    if (stm == 0) {
        std::memcpy(out.us, w_arr, sizeof(w_arr));
        std::memcpy(out.them, b_arr, sizeof(b_arr));
    } else {
        std::memcpy(out.us, b_arr, sizeof(b_arr));
        std::memcpy(out.them, w_arr, sizeof(w_arr));
    }

    bool is_quiet = has_quiet ? quiet_val : infer_is_quiet(parts[0], parts[1].empty() ? 'w' : static_cast<char>(std::tolower(static_cast<unsigned char>(parts[1][0]))), bm_uci, ev_raw, rho);
    out.eval_mask = (is_quiet && std::abs(ev_raw) < 9000.0) ? 1 : 0;
    double ev_pawns = std::clamp(ev_raw, -1000.0, 1000.0) / 100.0;
    out.eval_pawns = static_cast<float>(ev_pawns);
    if (has_wdl) out.wdl = static_cast<float>(wdl_raw);
    else if (has_gr) out.wdl = static_cast<float>(gr_raw);
    else out.wdl = static_cast<float>(1.0 / (1.0 + std::exp(-std::clamp(ev_pawns / WDL_SCALE_PAWNS, -20.0, 20.0))));

    out.tau = static_cast<float>(tau);
    out.rho = static_cast<float>(rho);
    out.rs = static_cast<float>(rs);
    out.bm_lbl = static_cast<uint16_t>(bm_lbl);
    out.sm_lbl = static_cast<uint16_t>(sm_lbl);
    return true;
}

static void write_npy(const fs::path& path, const char* descr, size_t n, size_t cols, const void* data, size_t elem_bytes) {
    std::string shape = (cols > 1) ? ("(" + std::to_string(n) + ", " + std::to_string(cols) + ")") : ("(" + std::to_string(n) + ",)");
    std::string hdr = "{'descr': '" + std::string(descr) + "', 'fortran_order': False, 'shape': " + shape + ", }";
    size_t pre = 10 + hdr.size() + 1;
    size_t pad = (64 - (pre % 64)) % 64;
    hdr.append(pad, ' ');
    hdr.push_back('\n');
    uint16_t hlen = static_cast<uint16_t>(hdr.size());
    uint8_t magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, static_cast<uint8_t>(hlen & 0xff), static_cast<uint8_t>(hlen >> 8)};
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(magic), 10);
    out.write(hdr.data(), hdr.size());
    if (n > 0) out.write(reinterpret_cast<const char*>(data), n * cols * elem_bytes);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: fast_pack <out_dir> <dedup:0|1> <file1> [file2...]\n";
        return 1;
    }
    fs::path out_dir(argv[1]);
    bool dedup = (std::atoi(argv[2]) != 0);
    std::vector<std::string> files;
    for (int i = 3; i < argc; ++i) files.emplace_back(argv[i]);

    unsigned n_threads = std::max(1u, std::min<unsigned>(8u, std::thread::hardware_concurrency()));
    std::vector<std::vector<Record>> thread_recs(n_threads);
    std::atomic<size_t> next_f{0};

    auto worker = [&](unsigned tid) {
        auto& vec = thread_recs[tid];
        std::string line;
        while (true) {
            size_t idx = next_f.fetch_add(1);
            if (idx >= files.size()) break;
            std::ifstream in(files[idx], std::ios::binary);
            if (!in) continue;
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                Record r{};
                if (parse_json_line(line.data(), line.size(), r)) {
                    vec.push_back(r);
                }
            }
        }
    };

    std::vector<std::thread> pool;
    for (unsigned t = 0; t < n_threads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    size_t raw_n = 0;
    for (const auto& v : thread_recs) raw_n += v.size();
    if (raw_n == 0) {
        std::cout << "0 0\n";
        return 0;
    }

    std::vector<Record> all;
    all.reserve(raw_n);
    for (auto& v : thread_recs) {
        all.insert(all.end(), v.begin(), v.end());
        std::vector<Record>().swap(v);
    }

    size_t uniq_n = raw_n;
    if (dedup) {
        std::vector<size_t> order(raw_n);
        for (size_t i = 0; i < raw_n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (all[a].hash != all[b].hash) return all[a].hash < all[b].hash;
            return a < b;
        });

        std::vector<Record> merged;
        merged.reserve(raw_n);
        size_t i = 0;
        while (i < raw_n) {
            size_t j = i + 1;
            uint64_t h = all[order[i]].hash;
            double s_ev = all[order[i]].eval_pawns, s_wdl = all[order[i]].wdl;
            double s_msk = all[order[i]].eval_mask, s_tau = all[order[i]].tau;
            double s_rho = all[order[i]].rho, s_rs = all[order[i]].rs;
            while (j < raw_n && all[order[j]].hash == h) {
                const Record& rj = all[order[j]];
                s_ev += rj.eval_pawns; s_wdl += rj.wdl;
                s_msk += rj.eval_mask; s_tau += rj.tau;
                s_rho += rj.rho; s_rs += rj.rs;
                ++j;
            }
            double cnt = static_cast<double>(j - i);
            Record r = all[order[i]];
            r.eval_pawns = static_cast<float>(s_ev / cnt);
            r.wdl = static_cast<float>(s_wdl / cnt);
            r.eval_mask = ((s_msk / cnt) >= 0.5) ? 1 : 0;
            r.tau = static_cast<float>(s_tau / cnt);
            r.rho = static_cast<float>(s_rho / cnt);
            r.rs = static_cast<float>(s_rs / cnt);
            merged.push_back(r);
            i = j;
        }
        all.swap(merged);
        uniq_n = all.size();
    }

    fs::create_directories(out_dir);
    std::vector<uint16_t> us_buf(uniq_n * 32), them_buf(uniq_n * 32);
    std::vector<float> ev_buf(uniq_n), wdl_buf(uniq_n), tau_buf(uniq_n), rho_buf(uniq_n), rs_buf(uniq_n);
    std::vector<uint8_t> msk_buf(uniq_n);
    std::vector<uint16_t> bm_buf(uniq_n), sm_buf(uniq_n);

    for (size_t i = 0; i < uniq_n; ++i) {
        const Record& r = all[i];
        std::memcpy(&us_buf[i * 32], r.us, 32 * sizeof(uint16_t));
        std::memcpy(&them_buf[i * 32], r.them, 32 * sizeof(uint16_t));
        ev_buf[i] = r.eval_pawns;
        wdl_buf[i] = r.wdl;
        msk_buf[i] = r.eval_mask;
        tau_buf[i] = r.tau;
        rho_buf[i] = r.rho;
        rs_buf[i] = r.rs;
        bm_buf[i] = r.bm_lbl;
        sm_buf[i] = r.sm_lbl;
    }

    write_npy(out_dir / "us_indices.npy", "<u2", uniq_n, 32, us_buf.data(), 2);
    write_npy(out_dir / "them_indices.npy", "<u2", uniq_n, 32, them_buf.data(), 2);
    write_npy(out_dir / "labels_eval.npy", "<f4", uniq_n, 1, ev_buf.data(), 4);
    write_npy(out_dir / "labels_wdl.npy", "<f4", uniq_n, 1, wdl_buf.data(), 4);
    write_npy(out_dir / "labels_eval_mask.npy", "|u1", uniq_n, 1, msk_buf.data(), 1);
    write_npy(out_dir / "labels_tau.npy", "<f4", uniq_n, 1, tau_buf.data(), 4);
    write_npy(out_dir / "labels_rho.npy", "<f4", uniq_n, 1, rho_buf.data(), 4);
    write_npy(out_dir / "labels_rs.npy", "<f4", uniq_n, 1, rs_buf.data(), 4);
    write_npy(out_dir / "best_move_label.npy", "<u2", uniq_n, 1, bm_buf.data(), 2);
    write_npy(out_dir / "sec_move_label.npy", "<u2", uniq_n, 1, sm_buf.data(), 2);

    std::cout << raw_n << " " << uniq_n << "\n";
    return 0;
}
