#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <chrono>
#include <unistd.h>
#include <memory>
#include <stdexcept>
#include <array>
#include <thread>
#include <sys/wait.h>
#include <signal.h>
#include <map>
#include <set>
#include "harenn_data.h"

/**
 * HARENN DATA GENERATION WITH STOCKFISH (PRODUCTION QUALITY)
 * Generates multi-head training data:
 * Eval, Tactical Complexity, MCS Map, Horizon Risk, Resolution Score.
 */

namespace Nextfish {

    // --- LỚP GIAO TIẾP VỚI STOCKFISH ---
    // Giao tiếp qua Pipe liên tục (Persistent Process) để tối ưu hiệu suất
    class Stockfish {
    private:
        int pid = -1;
        int pipe_in[2];  // Parent -> Child (Write to Stockfish)
        int pipe_out[2]; // Child -> Parent (Read from Stockfish)

    public:
        struct SearchResult {
            int score_cp = 0;
            std::string best_move_uci;
            std::vector<std::string> pv;
        };

        // Helper for MCS generation
        struct MoveInfo {
            std::string uci;
            int from;
            int to;
        };

        Stockfish(const std::string& path) {
            if (pipe(pipe_in) < 0 || pipe(pipe_out) < 0) {
                throw std::runtime_error("Failed to create pipes");
            }

            pid = fork();
            if (pid < 0) {
                throw std::runtime_error("Failed to fork process");
            }

            if (pid == 0) { // Child process
                // Redirect stdin/stdout
                dup2(pipe_in[0], STDIN_FILENO);
                dup2(pipe_out[1], STDOUT_FILENO);

                // Close unused ends
                close(pipe_in[0]);
                close(pipe_in[1]);
                close(pipe_out[0]);
                close(pipe_out[1]);

                // Execute Stockfish
                execl(path.c_str(), path.c_str(), nullptr);
                exit(1); // Should not reach here
            } else { // Parent process
                // Close unused ends
                close(pipe_in[0]);
                close(pipe_out[1]);

                // Initialize Engine
                write_cmd("uci");
                wait_for("uciok");
                write_cmd("isready");
                wait_for("readyok");
                write_cmd("ucinewgame");
            }
        }

        ~Stockfish() {
            if (pid > 0) {
                write_cmd("quit");
                close(pipe_in[1]);
                close(pipe_out[0]);
                waitpid(pid, nullptr, 0);
            }
        }

        void write_cmd(const std::string& cmd) {
            std::string full_cmd = cmd + "\n";
            write(pipe_in[1], full_cmd.c_str(), full_cmd.length());
        }

        // Đọc một dòng từ pipe
        std::string read_line() {
            std::string line = "";
            char c;
            while (read(pipe_out[0], &c, 1) > 0) {
                if (c == '\n') break;
                line += c;
            }
            return line;
        }

        // Đọc và bỏ qua cho đến khi gặp token
        void wait_for(const std::string& token) {
            while (true) {
                std::string line = read_line();
                if (line.find(token) != std::string::npos) break;
            }
        }

        // Search và trả về kết quả
        SearchResult search(const std::string& fen, int depth) {
            SearchResult result;
            
            write_cmd("position fen " + fen);
            write_cmd("go depth " + std::to_string(depth));

            while (true) {
                std::string line = read_line();
                std::stringstream line_ss(line);
                std::string token;
                line_ss >> token;

                if (token == "info") {
                    std::string key;
                    // Parse info line
                    while (line_ss >> key) {
                        if (key == "score") {
                            std::string type;
                            int value;
                            line_ss >> type >> value;
                            if (type == "cp") result.score_cp = value; // Score relative to side to move
                            else if (type == "mate") result.score_cp = (value > 0) ? 30000 - value : -30000 - value;
                        }
                    }
                } else if (token == "bestmove") {
                    line_ss >> result.best_move_uci;
                    break; // Search finished
                }
            }
            
            return result;
        }

        // Get legal moves using 'go perft 1'
        std::vector<MoveInfo> get_legal_moves(const std::string& fen) {
            std::vector<MoveInfo> moves;
            write_cmd("position fen " + fen);
            write_cmd("go perft 1");

            while (true) {
                std::string line = read_line();
                if (line.find("Nodes searched") != std::string::npos) break;
                
                // Line format: "e2e4: 1"
                size_t colon_pos = line.find(':');
                if (colon_pos != std::string::npos) {
                    std::string move_str = line.substr(0, colon_pos);
                    // Trim whitespace
                    move_str.erase(0, move_str.find_first_not_of(" \t"));
                    move_str.erase(move_str.find_last_not_of(" \t") + 1);
                    
                    if (move_str.length() >= 4) {
                        MoveInfo m;
                        m.uci = move_str;
                        m.from = (move_str[0] - 'a') + (move_str[1] - '1') * 8;
                        m.to   = (move_str[2] - 'a') + (move_str[3] - '1') * 8;
                        moves.push_back(m);
                    }
                }
            }
            return moves;
        }

        // Get current FEN from Stockfish using 'd' command
        std::string get_fen() {
            write_cmd("d");
            while (true) {
                std::string line = read_line();
                if (line.find("Fen: ") != std::string::npos) {
                    return line.substr(line.find("Fen: ") + 5);
                }
                if (line.find("Checkers:") != std::string::npos) break;
            }
            return "";
        }
    };

    // --- LỚP BÀN CỜ TỐI GIẢN ---
    // Chỉ dùng để theo dõi trạng thái và tạo FEN cho Stockfish
    class Board {
    public:
        int squares[64]; // 0: Empty, 1-6: White (P,N,B,R,Q,K), 7-12: Black
        int stm = 1; // 1: White, -1: Black
        std::string current_fen;
        
        Board() {
            reset();
        }

        void reset() {
            set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        }

        // Parse FEN to update board state
        void set_fen(const std::string& fen) {
            current_fen = fen;
            std::fill(std::begin(squares), std::end(squares), 0);
            
            std::stringstream ss(fen);
            std::string placement, side;
            ss >> placement >> side;
            
            stm = (side == "w") ? 1 : -1;

            int rank = 7, file = 0;
            for (char c : placement) {
                if (c == '/') { rank--; file = 0; }
                else if (isdigit(c)) { file += (c - '0'); }
                else {
                    const std::string pieces = "PNBRQKpnbrqk";
                    size_t idx = pieces.find(c);
                    if (idx != std::string::npos) {
                        squares[rank * 8 + file] = idx + 1;
                    }
                    file++;
                }
            }
        }

        // Tạo FEN string từ trạng thái bàn cờ hiện tại
        std::string get_fen() const {
            return current_fen;
        }

        uint64_t get_occupancy() { 
            uint64_t occ = 0;
            for(int i=0; i<64; ++i) if (squares[i] != 0) occ |= (1ULL << i);
            return occ;
        }
    };

    // --- THUẬT TOÁN HARENN CHUYÊN SÂU ---

    float calculate_std_dev(const std::vector<int>& values) {
        if (values.size() < 2) return 0.0f;
        float sum = std::accumulate(values.begin(), values.end(), 0.0f);
        float mean = sum / values.size();
        float sq_sum = 0;
        for (int v : values) sq_sum += (v - mean) * (v - mean);
        return std::sqrt(sq_sum / values.size());
    }

    float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    float clamp(float v, float lo, float hi) {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }

    // --- SINH NHÃN HARENN SỬ DỤNG STOCKFISH ---
    // Đây là nơi triển khai logic từ tài liệu HARENN của bạn
    HARENNEntry compute_full_harenn_labels(Board& board, Stockfish& sf) {
        HARENNEntry entry;
        std::string fen = board.get_fen();

        // --- Head 1: Evaluation (Score) ---
        // Search ở độ sâu tiêu chuẩn (ví dụ: 10)
        int main_depth = 10; 
        auto main_result = sf.search(fen, main_depth);
        int main_score = main_result.score_cp;

        // --- Head 2: Tactical Complexity (tau) ---
        // Phân tích sự biến động của điểm số ở các độ sâu khác nhau
        std::vector<int> depth_scores;
        std::vector<std::string> depth_moves;
        
        // Sample depths: 6, 8, 10
        // Tối ưu hóa: Chạy search cho các độ sâu phụ trước
        for (int d = 6; d < main_depth; d += 2) {
            auto res = sf.search(fen, d);
            depth_scores.push_back(res.score_cp);
            depth_moves.push_back(res.best_move_uci);
        }
        // Thêm kết quả của lần search chính (đã có sẵn) để tránh search lại
        // main_depth = 10.
        // Việc này giúp tiết kiệm 1 lần gọi search tốn kém.
        depth_scores.push_back(main_score);
        depth_moves.push_back(main_result.best_move_uci);

        // Component 1: Score Volatility
        float volatility = 0.0f;
        if (depth_scores.size() >= 2) {
            float sum_deltas = 0;
            for(size_t i=1; i<depth_scores.size(); ++i) {
                sum_deltas += std::abs(depth_scores[i] - depth_scores[i-1]);
            }
            float avg_delta = sum_deltas / (depth_scores.size() - 1);
            volatility = sigmoid((avg_delta - 15.0f) / 20.0f);
        }

        // Component 2: Best Move Instability
        int unique_moves = 0;
        if (!depth_moves.empty()) {
            std::set<std::string> distinct(depth_moves.begin(), depth_moves.end());
            unique_moves = distinct.size();
        }
        float instability = (unique_moves > 1) ? 0.5f : 0.0f;
        if (unique_moves > 2) instability = 1.0f;

        float tau = clamp(volatility * 0.6f + instability * 0.4f, 0.0f, 1.0f);
        entry.complexity_fixed = (int16_t)(tau * 100);

        // --- Head 3: Move Criticality Scores (MCS) ---
        // Logic: Compare full-depth score vs reduced-depth score for moves
        std::fill(std::begin(entry.mcs_map), std::end(entry.mcs_map), 0);
        
        auto legal_moves = sf.get_legal_moves(fen);
        
        // Optimization: Sample moves to save time (Top 5 + 40% of others)
        // int moves_processed = 0; // Uncomment to limit moves if too slow
        int mcs_depth_high = 7;
        int mcs_depth_low = 3;

        for (size_t i = 0; i < legal_moves.size(); ++i) {
            // Always process first 5 moves, then sample randomly
            if (i >= 5 && (rand() % 100 > 40)) continue;

            const auto& m = legal_moves[i];
            
            // Trick: Use UCI 'position ... moves ...' to search the resulting position
            // Note: Scores will be from opponent's perspective, but diff is absolute
            std::string pos_cmd = fen + " moves " + m.uci;
            
            int score_high = -sf.search(pos_cmd, mcs_depth_high).score_cp;
            int score_low = -sf.search(pos_cmd, mcs_depth_low).score_cp;

            int diff = std::abs(score_high - score_low);
            
            // Criticality formula from spec
            float crit = 0.0f;
            if (diff > 100) crit = 0.5f + std::min((float)diff/500.0f, 0.3f);
            else if (diff > 30) crit = 0.2f + (float)diff/200.0f;
            else crit = (float)diff/150.0f;
            
            crit = clamp(crit, 0.0f, 1.0f);
            
            // Map to 64x64 (from * 64 + to)
            if (m.from >= 0 && m.from < 64 && m.to >= 0 && m.to < 64) {
                entry.mcs_map[m.from * 64 + m.to] = (uint8_t)(crit * 255);
            }
            
            // moves_processed++;
            // if (moves_processed > 20) break; // Hard limit for speed
        }

        // --- Head 4: Horizon Risk (rho) ---
        // Compare Depth 10 vs Depth 13
        auto deep_result = sf.search(fen, 13);
        int deep_score = deep_result.score_cp;
        float rho = std::abs(main_score - deep_score) / 100.0f;
        if (main_result.best_move_uci != deep_result.best_move_uci) rho += 0.3f;
        entry.risk_fixed = (int16_t)(rho * 100);

        // --- Head 5: Resolution Score (rs) ---
        // Compare Static (Depth 1) vs QSearch Sim (Depth 4)
        auto static_res = sf.search(fen, 1);
        auto qs_res = sf.search(fen, 4);
        float rs = 1.0f - std::min(1.0f, std::abs(qs_res.score_cp - static_res.score_cp) / 150.0f);
        entry.resolution_fixed = (int16_t)(rs * 100);

        // Metadata & Board State
        entry.score = (int16_t)main_score;
        entry.stm = board.stm;    
        entry.occupancy = board.get_occupancy(); 
        
        int p_idx = 0;
        for(int i = 0; i < 64; ++i) {
            if ((entry.occupancy >> i) & 1) {
                if (p_idx < 32) {
                    entry.pieces[p_idx++] = (uint8_t)board.squares[i];
                }
            }
        }
        while(p_idx < 32) entry.pieces[p_idx++] = 0;
        
        return entry;
    }
}

// Hàm tải Opening Book từ file python sinh ra
std::vector<std::string> load_opening_book(const std::string& filename) {
    std::vector<std::string> book;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) book.push_back(line);
    }
    return book;
}

int play_one_game(std::ofstream& file, const std::string& opening_line, Nextfish::Stockfish& sf) {
    Nextfish::Board board;
    std::vector<HARENNEntry> game_positions;

    // Maintain UCI moves string to ensure correct game state (castling, en passant)
    std::string current_moves_uci = "";

    // 1. Parse opening moves
    std::stringstream ss(opening_line);
    std::string move;
    while (ss >> move) {
        if (!current_moves_uci.empty()) current_moves_uci += " ";
        current_moves_uci += move;
    }

    // 2. Vòng lặp Self-Play (Tự chơi đến hết ván)
    int move_count = 0;
    while (move_count < 200) { // Giới hạn 200 nước đi mỗi ván
        // Sync Board state with Stockfish
        if (current_moves_uci.empty()) sf.write_cmd("position startpos");
        else sf.write_cmd("position startpos moves " + current_moves_uci);
        
        // Get correct FEN (with castling rights) from Stockfish
        std::string fen = sf.get_fen();
        board.set_fen(fen);

        // Sinh dữ liệu cho vị trí hiện tại
        HARENNEntry entry = Nextfish::compute_full_harenn_labels(board, sf);
        game_positions.push_back(entry);

        // Engine tự chọn nước đi tốt nhất để tiếp tục ván cờ
        auto best_move_result = sf.search(board.get_fen(), 8); // Search depth 8 cho self-play
        std::string best_move_uci = best_move_result.best_move_uci;

        if (best_move_uci.empty() || best_move_uci == "(none)") break;
        
        if (!current_moves_uci.empty()) current_moves_uci += " ";
        current_moves_uci += best_move_uci;
        move_count++;
    }

    // 3. Gán kết quả hồi tố (Retroactive Result)
    int final_eval = sf.search(board.get_fen(), 10).score_cp;
    
    // SỬA LỖI: Chuyển điểm số tương đối (Stockfish) sang tuyệt đối (White perspective)
    // Nếu đến lượt Đen (stm = -1), đảo dấu điểm số
    int absolute_score = (board.stm == 1) ? final_eval : -final_eval;
    int game_result = (absolute_score > 100) ? 1 : (absolute_score < -100 ? -1 : 0);
    
    for (auto& entry : game_positions) {
        entry.result = game_result; // Cập nhật nhãn thắng/thua/hòa thực tế
        file.write(reinterpret_cast<const char*>(&entry), sizeof(HARENNEntry));
    }
    file.flush(); // Đảm bảo dữ liệu được ghi xuống đĩa ngay
    return game_result;
}

void run_generation(int games, std::string filename, Nextfish::Stockfish& sf) {
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Lỗi: Không thể mở file " << filename << std::endl;
        return;
    }

    // Tải Opening Book đã được sinh ra bởi preprocess_pgn.py
    auto opening_book = load_opening_book("book_moves.txt");
    if (opening_book.empty()) {
        opening_book.push_back(""); // Fallback nếu không có book
    }

    for (int i = 0; i < games; ++i) {
        // Chọn ngẫu nhiên một dòng khai cuộc
        std::string opening = opening_book[rand() % opening_book.size()];
        
        int result = play_one_game(file, opening, sf);
        std::string res_str = (result == 1) ? "1-0" : (result == -1 ? "0-1" : "1/2-1/2");
        
        std::cout << "Ván " << i + 1 << "/" << games << " | Kết quả: " << res_str << " | Ghi dữ liệu thành công." << std::endl;
    }
    file.close();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Sử dụng: ./harenn_gen <games> <output_file>" << std::endl;
        return 1;
    }
    
    unsigned int seed = (unsigned int)std::chrono::high_resolution_clock::now().time_since_epoch().count() + (unsigned int)getpid();
    srand(seed);
    
    std::cout << "Process ID: " << getpid() << " - Random Seed: " << seed << std::endl;
    std::cout << "--- HARENN Data Generation (Stockfish Backend) ---" << std::endl;

    int games = std::stoi(argv[1]);
    std::string output_file = argv[2];

    try {
        // THAY ĐỔI ĐƯỜNG DẪN NÀY tới file thực thi Stockfish của bạn
        Nextfish::Stockfish sf("./stockfish"); 

        std::cout << "Bắt đầu Self-Play và sinh dữ liệu HARENN Multi-Head..." << std::endl;
        run_generation(games, output_file, sf);
    } catch (const std::exception& e) {
        std::cerr << "Lỗi nghiêm trọng: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}