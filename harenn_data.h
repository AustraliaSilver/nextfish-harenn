#ifndef HARENN_DATA_H
#define HARENN_DATA_H

#include <cstdint>

// Đảm bảo không có padding giữa các trường dữ liệu để tối ưu dung lượng
#pragma pack(push, 1)

struct HARENNEntry {
    // --- Phần 1: Bàn cờ (41 bytes) ---
    uint64_t occupancy;      // 8 bytes: Vị trí các quân cờ
    uint8_t pieces[32];      // 32 bytes: Loại quân (nén)
    int8_t stm;              // 1 byte: Side to move (1: Trắng, -1: Đen)

    // --- Phần 2: Các nhãn Đa nhiệm (Multi-task Labels) ---
    // Head 1: Evaluation (3 bytes)
    int16_t score;           // 2 bytes: Điểm đánh giá (centipawns)
    int8_t result;           // 1 byte: Kết quả (1: Thắng, 0: Hòa, -1: Thua)

    // Head 2: Tactical Complexity (2 bytes)
    // Lưu giá trị thực * 100 vào số nguyên 16-bit
    int16_t complexity_fixed; 

    // Head 3: Move Criticality Scores (64 bytes)
    // Bản đồ 64x64 mô tả tầm quan trọng của từng cặp (from, to)
    // Cập nhật theo tài liệu HARENN Architecture 2.3
    uint8_t mcs_map[64 * 64]; // 4096 bytes

    // Head 4 & 5: Horizon Risk & Resolution (4 bytes)
    // Lưu giá trị thực * 100
    int16_t risk_fixed;
    int16_t resolution_fixed;
};

#pragma pack(pop)

#endif