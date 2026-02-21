#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "harenn_data.h"

/**
 * HARENN FULL LOGIC INTEGRATION
 * * Lưu ý: Tệp này giả định sự tồn tại của các hàm cơ bản từ Engine Nextfish.
 * Bạn cần include các header thực tế của engine (ví dụ: position.h, search.h)
 */

namespace Nextfish {

    // --- CÁC HÀM GIẢ ĐỊNH TỪ ENGINE (CẦN THAY THẾ BẰNG HÀM THẬT) ---
    
    // Giả lập hàm lấy Static Evaluation (Đánh giá tĩnh)
    int get_static_eval() {
        return (rand() % 100) - 50; 
    }
    
    // Giả lập hàm Search chính trả về Score
    int run_search(int nodes, int depth) { 
        // Trong thực tế: return Search::think(nodes, depth);
        return 45 + (rand() % 20); 
    }

    // Giả lập danh sách nước đi hợp lệ cho vị trí hiện tại
    struct MockMove { int from, to; };
    std::vector<MockMove> get_legal_moves() {
        std::vector<MockMove> moves;
        for(int i=0; i<5; ++i) moves.push_back({rand()%64, rand()%64});
        return moves;
    }

    // --- LOGIC HARENN CỐT LÕI ---

    // Hàm tính độ lệch chuẩn để đo mức độ hỗn loạn của thế trận
    float calculate_std_dev(const std::vector<int>& values) {
        if (values.empty()) return 0.0f;
        float sum = std::accumulate(values.begin(), values.end(), 0.0);
        float mean = sum / values.size();
        float sq_sum = 0;
        for (int v : values) sq_sum += (v - mean) * (v - mean);
        return std::sqrt(sq_sum / values.size());
    }

    HARENNEntry compute_full_harenn_labels(int nodes) {
        HARENNEntry entry;
        
        // 1. Main Search: Lấy điểm số gốc làm tham chiếu
        int main_depth = 10;
        int main_score = run_search(nodes, main_depth); 

        // 2. Head 2: Tactical Complexity (τ)
        // Đo lường sự thay đổi của Score qua các Depth (6, 8, 10)
        // Nếu score nhảy vọt -> Thế trận có tính chiến thuật cao
        std::vector<int> multi_depth_scores;
        multi_depth_scores.push_back(run_search(nodes / 4, main_depth - 4));
        multi_depth_scores.push_back(run_search(nodes / 2, main_depth - 2));
        multi_depth_scores.push_back(main_score);
        
        float tau = calculate_std_dev(multi_depth_scores);
        // Sửa lỗi: tactical_complexity_fixed -> complexity_fixed (theo harenn_data.h)
        entry.complexity_fixed = (int16_t)(tau * 100);

        // 3. Head 3: Move Criticality Scores (MCS)
        // Đây là trái tim của HARENN: Xác định nước đi nào "nhạy cảm" nhất
        std::fill(entry.mcs_map, entry.mcs_map + 64, 0);
        auto moves = get_legal_moves();
        
        for (auto& m : moves) {
            // "Probe Search": Thử đi nước m và search với depth cực thấp (ví dụ D-4)
            // Nếu điểm số sụt giảm mạnh so với main_score -> Đây là nước đi duy nhất/quan trọng
            int reduced_score = run_search(nodes / 10, main_depth - 6);
            int criticality = std::abs(main_score - reduced_score);
            
            // Lưu vào map tại ô đích của nước đi
            // Giá trị càng cao, nước đi càng quan trọng cho việc tính toán Reduction
            entry.mcs_map[m.to % 64] = (uint8_t)std::min(255, criticality);
        }

        // 4. Head 4: Horizon Risk (ρ)
        // Độ lệch giữa Static Eval và Search Score (dấu hiệu của bẫy hoặc phối hợp)
        int static_eval = get_static_eval();
        float rho = std::abs(main_score - static_eval) / 100.0f;
        // Sửa lỗi: horizon_risk_fixed -> risk_fixed (theo định nghĩa mới trong harenn_data.h)
        entry.risk_fixed = (int16_t)(rho * 100);

        // 5. Head 5: Resolution Score (rs)
        // Khả năng "hội tụ" của search (ví dụ: tỷ lệ các nước đi bị loại bỏ sớm)
        float resolution = 0.85f; // Tạm thời để hằng số hoặc tính từ Branching Factor
        // Sửa lỗi: resolution_score_fixed -> resolution_fixed (theo harenn_data.h)
        entry.resolution_fixed = (int16_t)(resolution * 100);

        // Metadata & Board State
        entry.score = (int16_t)main_score;
        entry.result = 0; // Cập nhật sau khi kết thúc ván
        entry.stm = 1;    // White to move
        entry.occupancy = 0x0000FFFFFFFF0000ULL; // Cần lấy từ Board::get_occ()
        
        // Khởi tạo pieces để tránh dữ liệu rác
        for(int i = 0; i < 32; ++i) entry.pieces[i] = 0;
        
        return entry;
    }
}

void run_generation(int games, int nodes, std::string filename) {
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Khong the mo file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < games; ++i) {
        // Thực hiện đầy đủ các bước tính toán đa nhiệm
        HARENNEntry entry = Nextfish::compute_full_harenn_labels(nodes);
        
        file.write(reinterpret_cast<const char*>(&entry), sizeof(HARENNEntry));
        
        if (i % 50 == 0) {
            std::cout << "Generated " << i << " multi-head HARENN positions..." << std::endl;
        }
    }
    file.close();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: ./harenn_gen <nodes> <games> <output_file>" << std::endl;
        return 1;
    }

    int nodes = std::stoi(argv[1]);
    int games = std::stoi(argv[2]);
    std::string output_file = argv[3];

    std::cout << "Starting HARENN Production Engine..." << std::endl;
    run_generation(games, nodes, output_file);

    return 0;
}