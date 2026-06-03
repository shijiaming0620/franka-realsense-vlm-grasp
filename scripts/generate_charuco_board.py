import yaml
import cv2
from pathlib import Path

cfg = yaml.safe_load(open("config/charuco_board.yaml"))

squares_x = int(cfg["squares_x"])
squares_y = int(cfg["squares_y"])
square_length = float(cfg["square_length"])
marker_length = float(cfg["marker_length"])

dict_name = cfg.get("dictionary", "DICT_4X4_50")
dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

# OpenCV 4.5.4 用 CharucoBoard_create
if hasattr(cv2.aruco, "CharucoBoard_create"):
    board = cv2.aruco.CharucoBoard_create(
        squares_x,
        squares_y,
        square_length,
        marker_length,
        dictionary
    )
else:
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        dictionary
    )

px_per_square = 200
img_size = (squares_x * px_per_square, squares_y * px_per_square)

if hasattr(board, "draw"):
    img = board.draw(img_size)
else:
    img = board.generateImage(img_size)

out = Path("data/charuco_calib/charuco_7x5.png")
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), img)

print("Saved:", out)
print("Print this image at 100% scale.")
print("Expected square length:", square_length, "m =", square_length * 1000, "mm")
