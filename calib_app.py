import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.widgets import TextBox, Button, Slider
import cv2

DEFAULTS = [
    "30, 170, 170",
    "5, 100, 100",
    "128, 170, 170",
    "10, 100, 100",
]

plt.ion()
fig = plt.figure(layout="constrained", figsize=(9, 4.5))
subfigs = fig.subfigures(1, 2)
subfigs[0].set_facecolor('0.75')

# gs = gridspec.GridSpec(1, 2,height_ratios=[5,1])

axs_left = subfigs[0].subplots(5, 1, )
subfigs_right = subfigs[1].subfigures(2, 1, )  # gridspec_kw={'height_ratios': [5, 1]}
ax_img_buttons = subfigs_right[1].subplots(1, 2)
axs_images = subfigs_right[0].subplots(1, 3)

images = [img for img in os.listdir("calib_img") if "positive" in img]
img_picker_prev = Button(ax_img_buttons[1], "Prev")
img_picker_next = Button(ax_img_buttons[0], "Next")
axs_images[0].imshow(plt.imread("img.png"))

textbox_states = [
    TextBox(axs_left[0], "Yellow HSV"),
    TextBox(axs_left[1], "Yellow Ranges"),
    TextBox(axs_left[2], "Purple HSV"),
    TextBox(axs_left[3], "Purple Ranges"),
]
for textbox_state, default in zip(textbox_states, DEFAULTS):
    textbox_state.set_val(default)

submit_button = (Button(axs_left[4], "Submit"))

a = 2

for axs_image in axs_images:
    axs_image.axis("off")


def get_mask(fname, lower, upper) -> np.ndarray:
    image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(image, lower, upper)
    return mask


def submit(event):
    change_image(img_idx)


def check_valid_input(str):
    try:
        arr = np.array(list(map(int, textbox_states[0].text.split(","))))
    except ValueError:
        return False
    if len(arr) != 3:
        return False
    return True


img_idx = 0
def change_image(val):
    global img_idx
    img_idx = val
    print(f"Showing img: {img_idx}")

    hsvs = []
    ranges = []
    for i in range(2):
        hsv = np.array(list(map(int, textbox_states[2 * i].text.split(","))))
        color_range = np.array(list(map(int, textbox_states[2 * i + 1].text.split(","))))
        if check_valid_input(hsv):
            hsvs.append(hsv)
        else:
            hsvs.append(DEFAULTS[2 * i])

        if check_valid_input(color_range):
            ranges.append(color_range)
        else:
            ranges.append(DEFAULTS[2 * i + 1])
    lowers = []
    uppers = []
    for hsv, color_range in zip(hsvs, ranges):
        lower = hsv - color_range
        upper = hsv + color_range
        lowers.append(lower)
        uppers.append(upper)

    print(f"{lowers=} {uppers=}")
    axs_images[0].imshow(cv2.cvtColor(cv2.imread(f"calib_img/{images[int(val)]}"), cv2.COLOR_RGB2HSV))
    axs_images[1].imshow(
        get_mask(f"calib_img/{images[int(val)]}", lowers[0], uppers[0])
    )
    axs_images[2].imshow(
        get_mask(f"calib_img/{images[int(val)]}", lowers[1], uppers[1])
    )
    fig.canvas.draw_idle()


submit_button.on_clicked(submit)
img_picker_prev.on_clicked(lambda event: change_image(img_idx - 1))
img_picker_next.on_clicked(lambda event: change_image(img_idx + 1))
plt.show()

while True:
    plt.pause(0.01)
