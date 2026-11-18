#include "harenn_datagen.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "bitboard.h"
#include "engine.h"
#include "movegen.h"
#include "position.h"
#include "search.h"
#include "types.h"
#include "uci.h"

namespace Stockfish {

namespace {

struct DatagenConfig {
    int games = 100;
    uint64_t nodes = 8192;
    int hash_mb = 64;
    int concurrency = 4;
    int multipv = 1;
    bool soft_nodes = true;
    int time_budget_sec = 0;
    std::string output_dir = "data";
    std::string worker_id = "";
    std::string book_path = "";
    int book_random_plies = 2;
    int branch_plies = 12;
    int branch_cp = 75;
};

std::vector<std::string> load_book_fens(const std::string& path) {
    std::vector<std::string> fens;
    if (path.empty()) return fens;
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Warning: could not open book file: " << path << std::endl;
        return fens;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string board, stm, castling, ep, hm, fm;
        if (!(ls >> board >> stm >> castling >> ep)) continue;
        if (ls >> hm >> fm && std::isdigit(static_cast<unsigned char>(hm[0])) &&
            std::isdigit(static_cast<unsigned char>(fm[0]))) {
            fens.push_back(board + " " + stm + " " + castling + " " + ep + " " + hm + " " + fm);
        } else {
            fens.push_back(board + " " + stm + " " + castling + " " + ep + " 0 1");
        }
    }
    return fens;
}

struct RawPosRecord {
    std::string fen;
    int stm = 0;
    int eval_score = 0;
    int depth = 1;
    std::string best_move;
    int best_move_label = 0;
    std::string sec_move;
    int sec_move_label = 0;
    int material = 0;
    int piece_count = 0;
    double tau = 0.0;
    double rho = 0.0;
    double rs = 0.5;
    bool is_quiet = true;
};

int score_to_cp(const Score& s) {
    if (s.is<Score::InternalUnits>()) {
        return s.get<Score::InternalUnits>().value;
    }
    if (s.is<Score::Mate>()) {
        int p = s.get<Score::Mate>().plies;
        return (p > 0) ? (10000 - p) : (-10000 - p);
    }
    if (s.is<Score::Tablebase>()) {
        auto tb = s.get<Score::Tablebase>();
        return (tb.win ? 20000 : -20000) - tb.plies;
    }
    return 0;
}

int move_to_label(const Position& pos, Move m) {
    if (m == Move::none() || m == Move::null()) return 0;
    Piece pc = pos.piece_on(m.from_sq());
    if (pc == NO_PIECE) return 0;
    PieceType moved_pt = (m.type_of() == PROMOTION) ? m.promotion_type() : type_of(pc);
    int pt = static_cast<int>(moved_pt) - 1;
    int c = (color_of(pc) == BLACK) ? 6 : 0;
    int p_idx = pt + c;
    int sq = static_cast<int>(m.to_sq()) ^ 56;
    return std::min(sq * 12 + p_idx, 767);
}

int count_material(const Position& pos) {
    int w = pos.count<PAWN>(WHITE) * 1 + pos.count<KNIGHT>(WHITE) * 3 +
            pos.count<BISHOP>(WHITE) * 3 + pos.count<ROOK>(WHITE) * 5 +
            pos.count<QUEEN>(WHITE) * 9;
    int b = pos.count<PAWN>(BLACK) * 1 + pos.count<KNIGHT>(BLACK) * 3 +
            pos.count<BISHOP>(BLACK) * 3 + pos.count<ROOK>(BLACK) * 5 +
            pos.count<QUEEN>(BLACK) * 9;
    return w - b;
}

bool is_insufficient_material(const Position& pos) {
    if (pos.count<PAWN>() || pos.count<ROOK>() || pos.count<QUEEN>())
        return false;
    int total_minors = pos.count<KNIGHT>() + pos.count<BISHOP>();
    if (total_minors <= 1)
        return true; // KvK, KNvK, KBvK
    if (pos.count<KNIGHT>() == 0 && pos.count<BISHOP>(WHITE) == 1 && pos.count<BISHOP>(BLACK) == 1) {
        constexpr Bitboard DARK_SQUARES = 0xAA55AA55AA55AA55ULL;
        bool w_dark = (pos.pieces(WHITE, BISHOP) & DARK_SQUARES) != 0;
        bool b_dark = (pos.pieces(BLACK, BISHOP) & DARK_SQUARES) != 0;
        if (w_dark == b_dark)
            return true; // KBvKB same-color bishops
    }
    return false;
}

std::string generate_dfrc_fen(std::mt19937_64& rng) {
    auto gen_rank = [&](std::mt19937_64& r) {
        char rank[8] = {0};
        std::uniform_int_distribution<int> d4(0, 3);
        rank[d4(r) * 2] = 'B';
        rank[d4(r) * 2 + 1] = 'B';
        auto place_empty = [&](char ch) {
            std::vector<int> emp;
            for (int i = 0; i < 8; ++i) if (rank[i] == 0) emp.push_back(i);
            std::uniform_int_distribution<size_t> d(0, emp.size() - 1);
            rank[emp[d(r)]] = ch;
        };
        place_empty('Q');
        place_empty('N');
        place_empty('N');
        std::vector<int> emp;
        for (int i = 0; i < 8; ++i) if (rank[i] == 0) emp.push_back(i);
        rank[emp[0]] = 'R';
        rank[emp[1]] = 'K';
        rank[emp[2]] = 'R';
        return std::string(rank, 8);
    };

    std::string wr = gen_rank(rng);
    std::string br = gen_rank(rng);
    std::string bl = br;
    for (char& c : bl) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    std::string castling;
    for (int i = 0; i < 8; ++i) if (wr[i] == 'R') castling += static_cast<char>('A' + i);
    for (int i = 0; i < 8; ++i) if (br[i] == 'R') castling += static_cast<char>('a' + i);

    return bl + "/pppppppp/8/8/8/8/PPPPPPPP/" + wr + " w " + castling + " - 0 1";
}

std::string first_token(std::string_view sv) {
    while (!sv.empty() && sv.front() == ' ') sv.remove_prefix(1);
    size_t sp = sv.find(' ');
    if (sp == std::string_view::npos) return std::string(sv);
    return std::string(sv.substr(0, sp));
}

void worker_fn(int tid, const DatagenConfig& cfg, int max_games,
               Engine& engine,
               const std::vector<std::string>& book_fens,
               std::atomic<uint64_t>& total_positions,
               std::chrono::steady_clock::time_point start_tp,
               std::mutex& log_mtx) {
    std::mt19937_64 rng(std::random_device{}() ^ (static_cast<uint64_t>(tid + 1) * 0x9E3779B97F4A7C15ULL));

    std::filesystem::create_directories(cfg.output_dir);
    std::time_t now_t = std::time(nullptr);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &now_t);
#else
    localtime_r(&now_t, &tm_buf);
#endif
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y%m%d_%H%M%S", &tm_buf);
    std::string wtag = cfg.worker_id.empty() ? "" : ("w" + cfg.worker_id + "_");
    std::string fname = "hnn_" + wtag + "p" + std::to_string(tid) + "_" + ts_buf + ".jsonl";
    std::filesystem::path out_path = std::filesystem::path(cfg.output_dir) / fname;
    std::ofstream out_f(out_path, std::ios::app);

    engine.set_tt_size(static_cast<usize>(std::max(1, cfg.hash_mb)));
    auto set_opt = [&](const std::string& name, const std::string& val) {
        std::istringstream os("name " + name + " value " + val);
        engine.get_options().setoption(os);
    };
    set_opt("Threads", "1");
    int cur_multipv = cfg.multipv;
    set_opt("MultiPV", std::to_string(cur_multipv));

    std::string m_shallow_str, sec_shallow_str;
    int cp_shallow = 0;
    bool has_shallow = false;

    std::array<uint64_t, 3> trace_thresh = {
        std::max<uint64_t>(512, cfg.nodes / 16),
        std::max<uint64_t>(1024, cfg.nodes / 4),
        std::max<uint64_t>(2048, cfg.nodes / 2)
    };
    std::array<int, 3> cp_trace = {0, 0, 0};
    std::array<bool, 3> has_trace = {false, false, false};

    std::string m_deep_str, sec_deep_str;
    int cp_deep = 0, cp_second = 0, achieved_depth = 1;
    std::string best_move_uci;

    uint64_t nodes_shallow = trace_thresh[0];

    engine.set_on_iter([](const auto&) {});
    engine.set_on_update_no_moves([](const auto&) {});
    engine.set_on_start([]() {});
    engine.set_on_verify_network([](const auto&) {});
    engine.set_on_update_full([&](const Search::InfoFull& info) {
        std::string mv = first_token(info.pv);
        int sc = score_to_cp(info.score);
        if (info.nodes >= nodes_shallow) {
            if (!has_shallow && info.multiPV == 1 && !mv.empty()) {
                m_shallow_str = mv;
                cp_shallow = sc;
                has_shallow = true;
            } else if (sec_shallow_str.empty() && info.multiPV == 2 && !mv.empty()) {
                sec_shallow_str = mv;
            }
        }
        if (info.multiPV == 1 && !mv.empty()) {
            for (int k = 0; k < 3; ++k) {
                if (!has_trace[k] && info.nodes >= trace_thresh[k]) {
                    cp_trace[k] = sc;
                    has_trace[k] = true;
                }
            }
            m_deep_str = mv;
            cp_deep = sc;
            achieved_depth = info.depth;
            if (cfg.soft_nodes && info.nodes >= cfg.nodes) {
                engine.stop();
            }
        } else if (info.multiPV == 2 && !mv.empty()) {
            sec_deep_str = mv;
            cp_second = sc;
        }
    });
    engine.set_on_bestmove([&](std::string_view bm, std::string_view) {
        best_move_uci = std::string(bm);
    });

    std::uniform_real_distribution<double> u01(0.0, 1.0);

    int games_played = 0;
    while (max_games <= 0 || games_played < max_games) {
        if (cfg.time_budget_sec > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_tp).count();
            if (elapsed >= cfg.time_budget_sec) break;
        }

        games_played++;
        bool is_standard = (u01(rng) < 0.70);
        std::string start_fen;
        int random_plies = 2;
        if (is_standard) {
            if (!book_fens.empty()) {
                std::uniform_int_distribution<size_t> bdist(0, book_fens.size() - 1);
                start_fen = book_fens[bdist(rng)];
                random_plies = cfg.book_random_plies;
            } else {
                start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
                random_plies = 3;
            }
        } else {
            start_fen = generate_dfrc_fen(rng);
            random_plies = std::uniform_int_distribution<int>(4, 6)(rng);
        }

        set_opt("UCI_Chess960", is_standard ? "false" : "true");

        std::string anchor_fen = start_fen;
        std::vector<std::string> played_moves;
        std::vector<RawPosRecord> game_positions;
        int last_eval_white = 0;
        std::vector<int> recent_evals_white;
        bool abnormal_stop = false;
        int win_adj_cnt = 0;
        int win_adj_sign = 0;
        int draw_adj_cnt = 0;
        double adjudicated_outcome = -1.0;

        int move_num = 0;
        while (true) {
            move_num++;

            engine.set_position(anchor_fen, played_moves);
            const Position& pos = engine.position();
            if (pos.rule50_count() == 0 && !played_moves.empty()) {
                anchor_fen = engine.fen();
                played_moves.clear();
            }
            MoveList<ALL> legal(pos);
            if (legal.size() == 0 || pos.is_draw(0) || is_insufficient_material(pos)) {
                break;
            }

            if (move_num <= random_plies) {
                std::uniform_int_distribution<size_t> d(0, legal.size() - 1);
                Move rm = *(legal.begin() + d(rng));
                played_moves.push_back(UCIEngine::move(rm, !is_standard));
                continue;
            }

            // Reset per-search trackers
            m_shallow_str.clear();
            sec_shallow_str.clear();
            has_shallow = false;
            cp_shallow = 0;
            cp_trace.fill(0);
            has_trace.fill(false);
            m_deep_str.clear();
            sec_deep_str.clear();
            cp_deep = 0;
            cp_second = 0;
            achieved_depth = 1;
            best_move_uci.clear();

            int tactical_cnt = 0;
            for (const auto& m : legal) {
                if (pos.capture(m) || pos.gives_check(m) || m.type_of() == PROMOTION) {
                    tactical_cnt++;
                }
            }
            double tactical_density = (legal.size() > 0)
                ? std::clamp(static_cast<double>(tactical_cnt + (pos.checkers() ? 1 : 0)) /
                             static_cast<double>(legal.size()), 0.0, 1.0)
                : 0.0;

            bool in_branch_window = (cfg.branch_plies > 0 && move_num > random_plies &&
                                     move_num <= random_plies + cfg.branch_plies);
            int target_mpv = in_branch_window ? 2 : cfg.multipv;
            if (target_mpv != cur_multipv) {
                cur_multipv = target_mpv;
                set_opt("MultiPV", std::to_string(cur_multipv));
            }

            Search::LimitsType limits;
            limits.nodes = cfg.soft_nodes ? (cfg.nodes * 4) : cfg.nodes;
            engine.go(limits);
            engine.wait_for_search_finished();

            if (best_move_uci.empty() || best_move_uci == "(none)") {
                abnormal_stop = true;
                break;
            }
            Move best_m = UCIEngine::to_move(pos, best_move_uci);
            if (best_m == Move::none()) {
                abnormal_stop = true;
                break;
            }

            if (!has_shallow) {
                cp_shallow = cp_deep;
                m_shallow_str = m_deep_str.empty() ? best_move_uci : m_deep_str;
            }
            if (m_deep_str.empty()) m_deep_str = best_move_uci;
            if (sec_deep_str == best_move_uci || sec_deep_str == m_deep_str) {
                sec_deep_str.clear();
                cp_second = cp_deep;
            }
            if (sec_shallow_str.empty()) sec_shallow_str = sec_deep_str;
            if (cur_multipv == 1) cp_second = cp_deep;

            last_eval_white = (pos.side_to_move() == WHITE) ? cp_deep : -cp_deep;
            recent_evals_white.push_back(last_eval_white);
            if (recent_evals_white.size() > 4) {
                recent_evals_white.erase(recent_evals_white.begin());
            }

            int eval_swing = std::abs(cp_shallow - cp_deep);
            int trace_vals[4];
            int trace_n = 0;
            for (int k = 0; k < 3; ++k) {
                if (has_trace[k]) trace_vals[trace_n++] = cp_trace[k];
            }
            trace_vals[trace_n++] = cp_deep;

            double rho = 0.0;
            if (trace_n >= 2) {
                double mean = 0.0;
                for (int k = 0; k < trace_n; ++k) mean += trace_vals[k];
                mean /= trace_n;
                double var = 0.0;
                double step_sum = 0.0;
                for (int k = 0; k < trace_n; ++k) {
                    double d = trace_vals[k] - mean;
                    var += d * d;
                    if (k > 0) step_sum += std::abs(trace_vals[k] - trace_vals[k - 1]);
                }
                double std_dev = std::sqrt(var / trace_n);
                double avg_step = step_sum / (trace_n - 1);
                rho = std::clamp(0.5 * (std_dev / 120.0) + 0.5 * (avg_step / 120.0), 0.0, 1.0);
            } else {
                rho = std::min(eval_swing / 200.0, 1.0);
            }

            double move_instability = 0.0;
            if (m_shallow_str != m_deep_str) move_instability += 0.5;
            if (!sec_shallow_str.empty() && !sec_deep_str.empty() && sec_shallow_str != sec_deep_str) move_instability += 0.5;
            double tau = std::clamp(0.60 * move_instability + 0.40 * tactical_density, 0.0, 1.0);

            if (std::abs(cp_deep) < 30) {
                rho = std::min(rho + 0.15, 1.0);
                tau = std::min(tau + 0.15, 1.0);
            } else if (std::abs(cp_deep) < 80) {
                rho = std::min(rho + 0.05, 1.0);
            }
            if (pos.checkers()) {
                rho = std::min(rho + 0.10, 1.0);
                tau = std::min(tau + 0.10, 1.0);
            }

            bool best_same = (m_shallow_str == m_deep_str);
            bool sec_same = (!sec_shallow_str.empty() && !sec_deep_str.empty() && sec_shallow_str == sec_deep_str);
            double s_agreement = (best_same && sec_same) ? 1.0 : (best_same ? 0.75 : (sec_same ? 0.50 : 0.25));
            double s_stability = std::max(0.0, 1.0 - (eval_swing / 150.0));
            int margin = std::max(0, cp_deep - cp_second);
            double s_margin = !sec_deep_str.empty() ? std::min(1.0, margin / 80.0) : 0.50;
            double rs = std::clamp(0.45 * s_stability + 0.35 * s_agreement + 0.20 * s_margin, 0.05, 0.99);

            bool dest_attacked = (pos.attackers_to(best_m.to_sq(), pos.pieces() ^ best_m.from_sq()) &
                                  pos.pieces(~pos.side_to_move())) != 0;
            bool quiet = !pos.checkers() && !pos.gives_check(best_m) && !pos.capture(best_m) &&
                         best_m.type_of() != PROMOTION && best_m.type_of() != CASTLING &&
                         !dest_attacked && eval_swing <= 150 && std::abs(cp_deep) < 1000;

            RawPosRecord rec;
            rec.fen = engine.fen();
            rec.stm = (pos.side_to_move() == WHITE) ? 0 : 1;
            rec.eval_score = cp_deep;
            rec.depth = achieved_depth;
            rec.best_move = best_move_uci;
            rec.best_move_label = move_to_label(pos, best_m);
            rec.sec_move = sec_deep_str;
            if (!sec_deep_str.empty()) {
                Move sm = UCIEngine::to_move(pos, sec_deep_str);
                rec.sec_move_label = (sm != Move::none()) ? move_to_label(pos, sm) : 0;
            }
            rec.material = count_material(pos);
            rec.piece_count = popcount(pos.pieces());
            rec.tau = std::round(tau * 10000.0) / 10000.0;
            rec.rho = std::round(rho * 10000.0) / 10000.0;
            rec.rs = std::round(rs * 10000.0) / 10000.0;
            rec.is_quiet = quiet;

            std::string move_to_play = best_move_uci;
            if (in_branch_window && !sec_deep_str.empty() &&
                (cp_deep - cp_second) <= cfg.branch_cp && u01(rng) < 0.50) {
                move_to_play = sec_deep_str;
            }
            game_positions.push_back(std::move(rec));
            played_moves.push_back(move_to_play);

            if (std::abs(last_eval_white) >= 500) {
                int sign = (last_eval_white > 0) ? 1 : -1;
                win_adj_cnt = (sign == win_adj_sign) ? (win_adj_cnt + 1) : 1;
                win_adj_sign = sign;
                draw_adj_cnt = 0;
                if (win_adj_cnt >= 6) {
                    adjudicated_outcome = (win_adj_sign > 0) ? 1.0 : 0.0;
                    break;
                }
            } else if (move_num >= 80 && std::abs(last_eval_white) <= 15) {
                win_adj_cnt = 0;
                win_adj_sign = 0;
                draw_adj_cnt++;
                if (draw_adj_cnt >= 16) {
                    adjudicated_outcome = 0.5;
                    break;
                }
            } else {
                win_adj_cnt = 0;
                win_adj_sign = 0;
                draw_adj_cnt = 0;
            }
        }

        // Determine game outcome
        engine.set_position(anchor_fen, played_moves);
        const Position& end_pos = engine.position();
        MoveList<ALL> end_legal(end_pos);
        double white_outcome = 0.5;
        if (adjudicated_outcome >= 0.0) {
            white_outcome = adjudicated_outcome;
        } else if (end_legal.size() == 0) {
            if (end_pos.checkers()) {
                white_outcome = (end_pos.side_to_move() == WHITE) ? 0.0 : 1.0;
            } else {
                white_outcome = 0.5;
            }
        } else if (end_pos.is_draw(0) || is_insufficient_material(end_pos)) {
            white_outcome = 0.5;
        } else if (!abnormal_stop && !recent_evals_white.empty()) {
            double eval_sum = 0.0;
            for (int ev : recent_evals_white) eval_sum += ev;
            double avg_eval_white = eval_sum / static_cast<double>(recent_evals_white.size());
            if (avg_eval_white >= 200.0) white_outcome = 1.0;
            else if (avg_eval_white <= -200.0) white_outcome = 0.0;
            else white_outcome = 0.5;
        }

        if (!game_positions.empty()) {
            std::string buf;
            buf.reserve(game_positions.size() * 256);
            for (const auto& r : game_positions) {
                double stm_win = (r.stm == 0) ? white_outcome : (1.0 - white_outcome);
                int gr = (stm_win == 1.0) ? 2 : ((stm_win == 0.0) ? 0 : 1);
                buf += "{\"fen\":\"" + r.fen + "\",\"stm\":" + std::to_string(r.stm) +
                       ",\"eval_score\":" + std::to_string(r.eval_score) +
                       ",\"depth\":" + std::to_string(r.depth) +
                       ",\"best_move\":\"" + r.best_move + "\"" +
                       ",\"best_move_label\":" + std::to_string(r.best_move_label) +
                       ",\"sec_move\":\"" + r.sec_move + "\"" +
                       ",\"sec_move_label\":" + std::to_string(r.sec_move_label) +
                       ",\"game_result\":" + std::to_string(gr) +
                       ",\"wdl\":" + (stm_win == 1.0 ? "1.0" : (stm_win == 0.0 ? "0.0" : "0.5")) +
                       ",\"material\":" + std::to_string(r.material) +
                       ",\"piece_count\":" + std::to_string(r.piece_count) +
                       ",\"tau\":" + std::to_string(r.tau) +
                       ",\"rho\":" + std::to_string(r.rho) +
                       ",\"rs\":" + std::to_string(r.rs) +
                       ",\"is_quiet\":" + (r.is_quiet ? "true" : "false") + "}\n";
            }
            out_f << buf;
            out_f.flush();
        }

        uint64_t added = game_positions.size();
        uint64_t cur_tot = total_positions.fetch_add(added) + added;
        double sec = std::max(0.001, std::chrono::duration<double>(std::chrono::steady_clock::now() - start_tp).count());
        if (games_played % 5 == 0 || games_played == 1) {
            std::lock_guard<std::mutex> lk(log_mtx);
            std::cout << "[W" << cfg.worker_id << "-P" << tid << " | G" << games_played << "] +" << added
                      << " pos | Total: " << cur_tot << " pos (" << std::fixed << std::setprecision(1)
                      << (cur_tot / sec) << " pos/s)" << std::endl;
        }
    }
}

} // namespace

void run_harenn_datagen(std::istringstream& is, const std::optional<std::filesystem::path>& bin_path) {
    DatagenConfig cfg;
    auto parse_int = [&](const std::string& flag, int& out) -> bool {
        std::string val;
        if (!(is >> val)) {
            std::cerr << "Error: missing value for " << flag << std::endl;
            return false;
        }
        try {
            size_t idx = 0;
            int v = std::stoi(val, &idx);
            if (idx != val.size()) throw std::invalid_argument("trailing chars");
            out = v;
            return true;
        } catch (const std::exception&) {
            std::cerr << "Error: invalid integer for " << flag << ": " << val << std::endl;
            return false;
        }
    };
    auto parse_u64 = [&](const std::string& flag, uint64_t& out) -> bool {
        std::string val;
        if (!(is >> val)) {
            std::cerr << "Error: missing value for " << flag << std::endl;
            return false;
        }
        try {
            size_t idx = 0;
            uint64_t v = std::stoull(val, &idx);
            if (idx != val.size()) throw std::invalid_argument("trailing chars");
            out = v;
            return true;
        } catch (const std::exception&) {
            std::cerr << "Error: invalid uint64 for " << flag << ": " << val << std::endl;
            return false;
        }
    };

    std::string tok;
    while (is >> tok) {
        if (tok == "--games" || tok == "-g") { if (!parse_int(tok, cfg.games)) return; }
        else if (tok == "--nodes" || tok == "-n") { if (!parse_u64(tok, cfg.nodes)) return; }
        else if (tok == "--hash") { if (!parse_int(tok, cfg.hash_mb)) return; }
        else if (tok == "--concurrency" || tok == "-j") { if (!parse_int(tok, cfg.concurrency)) return; }
        else if (tok == "--multipv") {
            int mpv = cfg.multipv;
            if (!parse_int(tok, mpv)) return;
            cfg.multipv = std::clamp(mpv, 1, 2);
        }
        else if (tok == "--hard-nodes") cfg.soft_nodes = false;
        else if (tok == "--soft-nodes") cfg.soft_nodes = true;
        else if (tok == "--time-budget") { if (!parse_int(tok, cfg.time_budget_sec)) return; }
        else if (tok == "--output" || tok == "-o") {
            if (!(is >> cfg.output_dir)) { std::cerr << "Error: missing value for " << tok << std::endl; return; }
        }
        else if (tok == "--worker-id" || tok == "-w") {
            if (!(is >> cfg.worker_id)) { std::cerr << "Error: missing value for " << tok << std::endl; return; }
        }
        else if (tok == "--book") {
            if (!(is >> cfg.book_path)) { std::cerr << "Error: missing value for " << tok << std::endl; return; }
        }
        else if (tok == "--book-random-plies") {
            int v = 0; if (!parse_int(tok, v)) return;
            cfg.book_random_plies = std::max(0, v);
        }
        else if (tok == "--branch-plies") {
            int v = 0; if (!parse_int(tok, v)) return;
            cfg.branch_plies = std::max(0, v);
        }
        else if (tok == "--branch-cp") {
            int v = 0; if (!parse_int(tok, v)) return;
            cfg.branch_cp = std::max(0, v);
        }
        else {
            std::cerr << "Warning: unknown harenn_datagen option: " << tok << std::endl;
        }
    }

    int conc = std::max(1, cfg.concurrency);
    int active_conc = (cfg.games > 0) ? std::min(conc, cfg.games) : conc;
    int base_games = (cfg.games <= 0) ? 0 : (cfg.games / active_conc);
    int rem_games = (cfg.games <= 0) ? 0 : (cfg.games % active_conc);

    std::vector<std::string> book_fens = load_book_fens(cfg.book_path);
    std::cout << "=== HARENN C++ Native Datagen (Stockfish Latest) | Games: " << cfg.games
              << " | Nodes: " << cfg.nodes << " | Hash: " << cfg.hash_mb << "MB | Threads: "
              << active_conc << " | MultiPV: " << cfg.multipv
              << " | Book: " << book_fens.size() << " pos | RandPlies: " << cfg.book_random_plies
              << " | BranchPlies: " << cfg.branch_plies << " (cp<=" << cfg.branch_cp << ") ===" << std::endl;

    std::atomic<uint64_t> total_positions{0};
    auto start_tp = std::chrono::steady_clock::now();
    std::mutex log_mtx;

    std::vector<std::unique_ptr<Engine>> engines;
    engines.reserve(active_conc);
    for (int i = 0; i < active_conc; ++i) {
        engines.push_back(std::make_unique<Engine>(bin_path));
    }

    std::vector<std::thread> threads;
    threads.reserve(active_conc);
    for (int i = 0; i < active_conc; ++i) {
        int worker_games = (cfg.games <= 0) ? 0 : (base_games + (i < rem_games ? 1 : 0));
        threads.emplace_back(worker_fn, i, std::cref(cfg), worker_games, std::ref(*engines[i]),
                             std::cref(book_fens), std::ref(total_positions), start_tp, std::ref(log_mtx));
    }
    for (auto& t : threads) t.join();

    double sec = std::max(0.001, std::chrono::duration<double>(std::chrono::steady_clock::now() - start_tp).count());
    std::cout << "=== Completed! Total: " << total_positions.load() << " positions in "
              << std::fixed << std::setprecision(1) << sec << "s ("
              << (total_positions.load() / sec) << " pos/s) ===" << std::endl;
}

} // namespace Stockfish
