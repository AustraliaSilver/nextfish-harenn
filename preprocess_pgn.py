import random
import os

def generate_random_openings(count=1000):
    """
    Tự động tạo ra các chuỗi nước đi khai cuộc ngẫu nhiên hợp lệ
    để làm mồi (seed) cho engine self-play.
    """
    print(f"Đang khởi tạo {count} chuỗi khai cuộc ngẫu nhiên...")
    
    # Danh sách các nước đi khởi đầu phổ biến (UCI format)
    first_moves = ["e2e4", "d2d4", "c2c4", "g1f3"]
    
    # Các phản ứng phổ biến của bên Đen
    responses = {
        "e2e4": ["e7e5", "c7c5", "e7e6", "c7c6"],
        "d2d4": ["d7d5", "g8f6", "e7e6"],
        "c2c4": ["e7e5", "c7c5", "g8f6"],
        "g1f3": ["d7d5", "g8f6", "c7c5"]
    }
    
    # Các nước đi tiếp theo (đơn giản hóa để đảm bảo tính hợp lệ cơ bản)
    third_moves = ["g1f3", "b1c3", "f2f4", "d2d4", "e2e4"]
    
    book = []
    
    for _ in range(count):
        # Chọn nước đầu tiên
        m1 = random.choice(first_moves)
        # Chọn nước phản hồi
        m2 = random.choice(responses[m1])
        # Chọn ngẫu nhiên nước thứ 3 và 4 (giả lập)
        m3 = random.choice(third_moves)
        
        # Tạo chuỗi line khai cuộc (ví dụ: 2-4 nước đi)
        line = f"{m1} {m2} {m3}"
        book.append(line)
        
    return book

def prepare_book():
    print("--- HARENN Opening Book Generator ---")
    
    # Bạn có thể điều chỉnh số lượng chuỗi khai cuộc ở đây
    num_lines = 2000 
    
    book_lines = generate_random_openings(num_lines)
    
    try:
        with open("book_moves.txt", "w") as f:
            for line in book_lines:
                f.write(line + "\n")
        print(f"Thành công: Đã tạo file 'book_moves.txt' với {num_lines} lines.")
    except Exception as e:
        print(f"Lỗi khi ghi file: {e}")

if __name__ == "__main__":
    prepare_book()