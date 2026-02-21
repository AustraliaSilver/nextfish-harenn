#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include "harenn_data.h"

namespace Nextfish {
    // Cấu trúc tạm thời để chứa kết quả tính toán từ Search
    struct SearchLabels {
        int score;
        float complexity;
        uint8_t mcs[64];
        float risk;
        float resolution;
    };

    // QUAN TRỌNG: Bạn cần thay logic này bằng hàm Search thực tế của Engine
    SearchLabels compute_harenn_labels(int nodes) {
        SearchLabels out;
        out.score = (rand() % 100);
        out.complexity = 1.45f; // Ví dụ
        out.risk = 0.2f;
        out.resolution = 0.9f;
        for(int i=0; i<64; ++i) out.mcs[i] = (uint8_t)(rand() % 256);
        return out;
    }
}

void run_datagen(int games, int nodes, std::string filename) {
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    if (!file.is_open()) return;

    for (int i = 0; i < games; ++i) {
        HARENNEntry entry;
        auto labels = Nextfish::compute_harenn_labels(nodes);

        // Gán dữ liệu bàn cờ (giả lập)
        entry.occupancy = 0x1234567890ABCDEFULL;
        entry.stm = 1;

        // Gán và nén nhãn HARENN
        entry.score = (int16_t)labels.score;
        entry.result = 0; // Thường được cập nhật sau khi kết thúc ván
        entry.complexity_fixed = (int16_t)(labels.complexity * 100);
        entry.risk_fixed = (int16_t)(labels.risk * 100);
        entry.resolution_fixed = (int16_t)(labels.resolution * 100);
        std::copy(labels.mcs, labels.mcs + 64, entry.mcs_map);

        file.write(reinterpret_cast<const char*>(&entry), sizeof(HARENNEntry));
    }
    file.close();
    std::cout << "Hoàn thành batch sinh " << games << " ván." << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) return 1;
    run_datagen(std::stoi(argv[2]), std::stoi(argv[1]), argv[3]);
    return 0;
}