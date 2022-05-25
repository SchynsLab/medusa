import numpy as np
from PIL import Image
from imageio import get_reader, get_writer
from medusa.render import Renderer
from medusa.core3d import load_h5
from collections import namedtuple
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
        self.show_vid = False
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

    def activate_vid(self, t, t_start=0, fade_in=0.5):

        self.vid_p = Params(t, t_start, fade_in)       
        self.show_vid = True
        self.vid_params = {
            't_start': self._t2f(t),
            't_end': self._t2f(t_start + t),
            'alpha': np.linspace(0, 1, num=self._t2f(fade_in)),
            'counter': 0
        }

    def reset(self):
        self.show_vid = False
        self.vid_params = {
            't_start': 0,
            't_end': np.inf
        } 

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
        for tf in tqdm(range(frames), desc=name):
            
            bg = self.background.copy()
            
            if self.show_vid:
                if tf >= self.vid_params['t_start'] and tf < self.vid_params['t_end']:            
                    still = self.get_vid_frame(tf)
                    alpha = self.vid_params['alpha'][self.vid_params['counter']]
                    bg = Image.blend(bg, still, alpha=alpha)
                    bg.putalpha(255)
                    self.vid_params['counter'] += 1

            recon = self._renderer(self.recon.v[tf, ...], self.recon.f)
            img = self._renderer.alpha_blend(recon, np.array(bg)[..., :3], face_alpha=0.1)
            img = Image.fromarray(img)            
            #recon = Image.fromarray(recon)
            img = np.array(img).astype(np.uint8)
            self._writer.append_data(img)
            self.current += 1
            
    def close(self):
        self._reader.close()
        self._writer.close()


class Params:
    
    def __init__(self, t, t_start, fade_in):
        self.t = t
        self.t_start = t_start
        self.fade_in = fade_in
        self._convert()
        
    def _convert(self):
        pass


    
if __name__ == '__main__':
    
    vid = 'demos/pexels_2.mp4'
    recon = 'demos/pexels_2_mediapipe.h5'
    f_out = 'demos/teaser_test.gif'
            
    demo = Demo(vid, recon, f_out, scale=0.2)
    demo.activate_vid()
    demo.render(t=1)
    demo.reset()
    demo.close()
        
        
        