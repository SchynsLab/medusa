import logging
from datetime import datetime

import numpy as np
from imageio import get_reader, get_writer
from matplotlib.font_manager import FontManager
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import rescale
from tqdm import tqdm

from medusa.io import load_h5
from medusa.render import PyRenderer as Renderer

logging.basicConfig(
    format="%(asctime)s [%(levelname)-7.7s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[logging.StreamHandler()],
)

default_font = FontManager().defaultFont["ttf"]


class Demo:
    def __init__(
        self, vid, recon, f_out, scale=1, background_color=(0, 0, 0), loglevel="INFO"
    ):

        self.logger = logging.getLogger("demo")
        self.logger.setLevel(loglevel)
        self.vid = vid
        self._load_recon(recon)
        self.f_out = f_out
        self.scale = scale
        self.background_color = background_color
        self.fps = None  # set when calling `start`
        self.current_frame = 0
        self.vid_params = None
        self.recon_params = None
        self.txt_params = []

    def _load_recon(self, recon):
        self.recon = dict()
        if not isinstance(recon, dict):
            recon = dict(default=recon)

        for name, file in recon.items():
            self.logger.info(f"Loading recon with name '{name}'")
            self.recon[name] = load_h5(file)

        self._active_recon = name

    def start(self):

        recon = list(self.recon.values())[0]
        self.fps = recon.sf

        vid_size = (np.array(recon.img_size) * self.scale).round()
        self.to_crop = (vid_size % 16).astype(int)
        vid_size = (vid_size - self.to_crop).astype(int)

        self._reader = get_reader(self.vid)
        self._writer = get_writer(self.f_out, mode="I", fps=self.fps)
        self.logger.info(
            f"Opening {self.f_out} with size {vid_size} and " f"{self.fps:.2f} FPS"
        )
        self._vid_size = vid_size

        # Setup renderer
        self._renderer = Renderer(
            viewport=vid_size, camera_type="intrinsic", wireframe=True
        )
        self._bg = Image.new("RGBA", tuple(vid_size.tolist()), self.background_color)

    def _draw_vid(self):
        pass

    def _draw_recon(self):
        pass

    def _draw_txt(self):
        pass

    def show_vid(self, start, end, fade_in=None, fade_out=None):
        self.vid_params = dict(start=start, end=end, fade_in=fade_in, fade_out=fade_out)

    def show_recon(self, start, end, wireframe=True, fade_in=None, fade_out=None):
        self.recon_params = dict(
            start=start,
            end=end,
            wireframe=wireframe,
            fade_in=fade_in,
            fade_out=fade_out,
        )

    def show_txt(self, start, end, text, pos=(0, 0), fontsize=2, **kwargs):
        self.txt_params.append(
            dict(
                start=start,
                end=end,
                xy=[round(p * vs) for p, vs in zip(pos, self._vid_size)],
                text=text,
                fill=(255, 255, 255),
                font=ImageFont.truetype(default_font, fontsize),
                **kwargs,
            )
        )

    def _should_show(self, t, params):

        if params is None:
            return False

        if t >= params["start"] and t <= params["end"]:
            return True
        else:
            return False

    def _compute_alpha_for_fade(self, t, params):

        start = params["start"]
        end = params["end"]
        full_t = end - start
        fade_in = params.get("fade_in")
        fade_out = params.get("fade_out")

        if fade_in is None and fade_out is None:
            return 1

        if fade_in is not None:
            if t <= (fade_in + start):
                alpha = (t - start) / fade_in
                return alpha

        if fade_out is not None:
            if t >= (full_t - fade_out):
                alpha = (full_t - t) / fade_out
                return alpha

        return 1

    def _blend(self, bg, img, alpha=None):

        if alpha is None:
            alpha = np.array(img.getchannel("A")) / 255.0
            alpha = alpha[..., None]

        img = np.array(bg) * (1 - alpha) + np.array(img) * alpha
        img = Image.fromarray(img.astype(np.uint8), mode="RGBA")
        img.putalpha(255)
        return img

    def play(self, duration):

        n_frames = self._t2f(duration)
        self.logger.info(f"Playing for {n_frames} frames")

        desc = datetime.now().strftime("%Y-%m-%d %H:%M [INFO   ]")
        bar_format = "{desc}  {bar}  [frame {n_fmt}/{total_fmt}]"
        pbar = tqdm(range(n_frames), bar_format=bar_format, desc=desc)

        for i_frame in pbar:
            t = i_frame / self.fps

            canvas = self._bg.copy()
            # Determine canvas (background)
            if self._should_show(t, self.vid_params):
                vid_img = self._reader.get_data(i_frame + self.current_frame)

                if self.scale != 1:
                    vid_img = self._resize(vid_img, to_pil=True)

                alpha = self._compute_alpha_for_fade(t, self.vid_params)
                canvas = self._blend(canvas, vid_img, alpha=alpha)

            if self._should_show(t, self.recon_params):
                recon = self.recon[self._active_recon]
                recon_img = self._renderer(
                    recon.v[i_frame + self.current_frame, ...], recon.f
                )
                recon_img = Image.fromarray(recon_img, mode="RGBA")
                canvas = self._blend(canvas, recon_img)

            for txt_params in self.txt_params:
                if self._should_show(t, txt_params):
                    tmp_canvas = canvas.copy()
                    draw = ImageDraw.Draw(tmp_canvas)
                    params = {
                        k: v
                        for k, v in txt_params.items()
                        if k not in ["start", "end", "fade_in", "fade_out"]
                    }
                    draw.text(**params)
                    alpha = self._compute_alpha_for_fade(t, txt_params)
                    canvas = self._blend(canvas, tmp_canvas, alpha=alpha)

            canvas = np.array(canvas)
            self._writer.append_data(canvas)

        self.current_frame += n_frames

    def _resize(self, img, to_pil=True):
        img = rescale(
            img, self.scale, anti_aliasing=True, channel_axis=2, preserve_range=True
        ).astype(np.uint8)

        if self.to_crop[0] != 0:
            img = img[:, : -self.to_crop[0], ...]

        if self.to_crop[1] != 0:
            img = img[: -self.to_crop[1], ...]

        if to_pil:
            img = Image.fromarray(img, mode="RGB").convert("RGBA")

        return img

    def _t2f(self, t):
        return int(round(t * self.fps))

    def close(self):
        self._reader.close()
        self._writer.close()


if __name__ == "__main__":

    vid = "demos/pexels_2.mp4"
    recon = "demos/pexels_2_mediapipe.h5"
    f_out = "demos/teaser_test.mp4"
    length = 9

    demo = Demo(vid, recon, f_out, scale=0.25)
    demo.start()
    intro = "Medusa:\na Python package for 4D face analysis"
    demo.show_txt(
        start=0,
        end=1.5,
        text=intro,
        pos=(0.5, 0.5),
        align="center",
        anchor="ms",
        fontsize=15,
    )
    demo.show_vid(start=1.5, end=length, fade_in=1, fade_out=None)
    demo.show_txt(
        start=2,
        end=3,
        text="Medusa can",
        pos=(0.05, 0.05),
        fontsize=25,
        fade_in=0.5,
        fade_out=0.5,
    )
    demo.show_txt(start=2.5, end=3.5, text="reconstruct", pos=(0.05, 0.4), fontsize=20)
    demo.show_txt(start=3.5, end=4.5, text="preprocess", pos=(0.05, 0.4), fontsize=20)
    demo.show_txt(
        start=4.5,
        end=6.5,
        text="and analyze\n\nface images and videos",
        pos=(0.05, 0.4),
        fontsize=20,
    )

    demo.show_txt(
        start=6,
        end=7,
        text="And supports reconstruction models like",
        pos=(0.05, 0.05),
        fontsize=20,
        fade_in=0.5,
    )
    demo.show_txt(
        start=7,
        end=8,
        text="And supports reconstruction models like",
        pos=(0.05, 0.05),
        fontsize=20,
    )
    # demo.show_txt(start=8, end=9, text='\nand DECA/EMOCA', pos=(0.05, 0.05), fontsize=20, fade_in=0.5)

    demo.show_recon(start=2.5, end=length)
    demo.play(duration=length)
    demo.close()
