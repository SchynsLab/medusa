import cv2
import os
import matplotlib.pyplot as plt


def plot_shape(v, f, f_out=None, color='lightgray', overlay=None,
               f_type='flame', figsize=(10, 8)):
    """ Plot a shape given vertices and faces.
    
    OLD (and slow), DO NOT USE
    
    Parameters
    ----------
    v : ndarray
        Numpy array (n x 3) with 3D vertices
    f : ndarray
        Numpy array (m x 3) with faces
    figsize : tuple
        Tuple with figure size
    """

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    coll = ax.plot_trisurf(
        v[:, 0], v[:, 1], v[:, 2], triangles=f,
        color=color, antialiased=True, cmap=None
    )
    if overlay is not None:
        coll.set_array(overlay)

    if f_type == 'flame':
        ax.view_init(90, -90)
    else:
        ax.view_init(90, 90)

    ax.axis('off')

    if f_out is not None:
        fig.savefig(f_out)
        plt.close()
    else:
        return fig, ax


def images_to_mp4(images, f_out):

    if os.path.isfile(f_out):
        os.remove(f_out)

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f_out, fourcc, 12.5, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

