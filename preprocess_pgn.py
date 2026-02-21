import os

def prepare_book():
    print("Preparing opening book for HARENN generation...")
    # Tạo tệp book_moves.txt giả lập nếu chưa có
    # Trong thực tế, bạn có thể tải từ một nguồn PGN chất lượng cao
    moves = [
        "e2e4 e7e5 g1f3 b8c6",
        "d2d4 d7d5 c2c4 e7e6",
        "g1f3 d7d5 g2g3 g8f6"
    ]
    with open("book_moves.txt", "w") as f:
        for move in moves:
            f.write(move + "\n")
    print("Opening book ready.")

if __name__ == "__main__":
    prepare_book()