import numpy as np
from PIL import Image
from imageio import get_reader, get_writer
from medusa.render import Renderer
from medusa.core import load_h5
from collections import defaultdict
from tqdm import tqdm


class Demo:
    
    def __init__(self, vid, recon, f_out, scale=1, background_color=(0, 0, 0)):
        
        self.vid = vid
        self.recon = load_h5(recon)
        self.f_out = f_out
        self.scale = scale
        self.background_color = background_color
        self.fps = self.recon.sf
        self.current = 0
        self._queue = []
        self._configure()
        
    def _configure(self):
        
        self._reader = get_reader(self.vid)
        self._writer = get_writer(self.f_out, mode='I', fps=self.fps)
        self.size = (int(self.recon.img_size[0] * self.scale),
                     int(self.recon.img_size[1] * self.scale))
        self._renderer = Renderer(viewport=self.size, camera_type='intrinsic',
                                  wireframe=True)
        self.background = Image.new('RGBA', self.size, self.background_color)

    def add_txt(self, txt, t, t_start=None):
        
        self._queue['txt']['t'].append(t)
        self._queue['txt']['t_start'].append(t_start)
    
    def add_recon(self, t, t_start=None):
        pass

    def add_vid(self, t, t_start=None):
        pass 

    def _t2f(self, t):
        return int(round(t * self.fps))

    def get_vid_frame(self, i):
        idx = self.current + i
        img = self._reader.get_data(idx)
        img = Image.fromarray(img).convert('RGBA')
        
        if self.scale != 1:
            img = img.resize(self.size, resample=Image.Resampling.BILINEAR)

        return img

    def render(self, t, name='Render'):
        
        frames = self._t2f(t)
        for f in tqdm(range(frames), desc=name):
            
            bg = self.background.copy()
            
            #if f >= self.f_vid_start:
            still = self.get_vid_frame(f)
            bg = Image.blend(bg, still, alpha=1)
            bg.putalpha(255)

            recon = self._renderer(self.recon.v[f, ...], self.recon.f)
            img = self._renderer.alpha_blend(recon, np.array(bg)[..., :3], face_alpha=0.1)
            
            #recon = Image.fromarray(recon)
            #img = Image.alpha_composite(recon, bg)
            img = np.array(img).astype(np.uint8)
            self._writer.append_data(img)
            
    def close(self):
        self._reader.close()
        self._writer.close()

    
if __name__ == '__main__':
    
    vid = 'demos/pexels_2.mp4'
    recon = 'demos/pexels_2_mediapipe.h5'
    f_out = 'demos/teaser_test.gif'
            
    demo = Demo(vid, recon, f_out, scale=0.2)
    demo.render(t=1)
    demo.close()
        
        
        