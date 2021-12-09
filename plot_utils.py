import os
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import math
import os

def create_figs(results):
    #Create figs
    figs = []
    for i,r in enumerate(results):
        figs_ = []
        for j,rr in enumerate(r):
            smile = rr["smile"]
            index = rr["index"]
            node_weights = rr["weights"]
            edge_weights = rr["edge_weights"]
            f = draw_chem_activations(smile=smile, index=index, node_weights=node_weights, edge_weights=edge_weights, colorbar=False,
                                      node_radius=0.03, edge_radius=0.01, colormap="Blues", size=(300, 300),
                                      vmax=1.0, vmin=0.0, save_path=rr["method"])
            figs_.append(f)
        figs.append(figs_)

    for i,r in enumerate(results):
        figs_ = []
        for j,rr in enumerate(r):
            smile = rr["smile"]
            index = rr["index"]
            f = draw_chem_activations(smile=smile, index=index, node_weights=None, edge_weights=None, colorbar=False,
                                      node_radius=0.03, edge_radius=0.01, colormap="Blues", size=(300, 300),
                                      vmax=1.0, vmin=0.0, save_path='original')
            figs_.append(f)
        figs.append(figs_)
    return figs


def create_im_arrs(figs):
    #Create image arrays from figures
    ims_arr = []
    for i,f in enumerate(figs):
        ims_arr_ = []
        for j,ff in enumerate(f):
            X = np.array(ff.canvas.renderer._renderer)
            ims_arr_.append(X)
        ims_arr.append(ims_arr_)
    return ims_arr


def draw_atom_disks(mol, node_weights, edge_weights, node_radius=0.05, edge_radius=0.05, step=0.001):
    """
    Draw disks of fixed radius around each atom's coordinate.
    `node_weights` controls the color of each Atom.
    `edge_weights` controls the color of each Bond.
    """
    x = np.arange(0, 1, step)
    y = np.arange(0, 1, step)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    # for node importance
    for i, (c_x, c_y) in mol._atomPs.items():
        base = (X - c_x) ** 2 + (Y - c_y) ** 2
        circle_mask = (base < node_radius**2)
        circle = circle_mask.astype('float') * node_weights[i]
        Z += circle
    # for edge importance
    for i, (x_i, y_i) in mol._atomPs.items():
        for j, (x_j, y_j) in mol._atomPs.items():
            if edge_weights[i][j]>0:
                line_mask = get_point_line_distance(X, Y, x_i, y_i, x_j, y_j) < edge_radius
                x_mask = (X< max(x_i, x_j)+edge_radius) * (X> min(x_i, x_j)-edge_radius)
                y_mask = (Y< max(y_i, y_j)+edge_radius) * (Y>min(y_i, y_j)-edge_radius)
                mask = line_mask * x_mask * y_mask
                circle = mask.astype('float') * edge_weights[i][j]
                Z += circle

    return X, Y, Z

def get_point_line_distance(X, Y, x_i, y_i, x_j, y_j):
    point_x = X
    point_y = Y
    line_s_x = x_i
    line_s_y = y_i
    line_e_x = x_j
    line_e_y = y_j

    if line_e_x - line_s_x == 0:
        return np.abs(point_x - line_s_x)

    if line_e_y - line_s_y == 0:
        return np.abs(point_y - line_s_y)

    k = (line_e_y - line_s_y) / (line_e_x - line_s_x)

    b = line_s_y - k * line_s_x

    dis = np.abs(k * point_x - point_y + b) / np.sqrt(k*k + 1)
    return dis


def draw_chem_activations(smile, index, node_weights, edge_weights, colorbar=False, colormap = 'RdBu',
                          step=0.001, size=(250, 250), node_radius=0.025, edge_radius=0.025,
                          coord_scale=1.5, title=None, vmax=1.0, vmin=-1.0, save_path='temp'):
    """
    Draw scalar activations on each
    """
    mol = Chem.MolFromSmiles(smile)

    Chem.rdmolops.SanitizeMol(mol)

    cmap = plt.cm.get_cmap(colormap)
    fig = Draw.MolToMPL(mol, coordScale=coord_scale, size=size, **{}, )

    # draw node importance
    if node_weights is not None:
        ax = fig.axes[0]
        x, y, z = draw_atom_disks(mol, node_radius=node_radius, edge_radius=edge_radius, node_weights=node_weights, edge_weights=edge_weights, step=step)
        ax.imshow(z, cmap=cmap, interpolation='bilinear', origin='lower',
                      extent=(0, 1, 0, 1), vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        if colorbar:
            plt.colorbar(sm, fraction=0.035, pad=0.04)
        if title:
            plt.title(title)

    save_dir = 'figs/' + save_path

    # Check whether the path already exists or not
    isExist = os.path.exists(save_dir)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(save_dir)
        print('Creating new directory ' + save_dir + ' to save the explanation visualization results')

    plt.savefig(save_dir + '/' + str(index) + '.jpeg', bbox_inches='tight')
    return fig


def plot_image_grid(grid,
                    row_labels_left,
                    row_labels_right,
                    col_labels,
                    super_col_labels=None,
                    file_name=None,
                    dpi=224,
                    c=10,
                    fontsize=24,
                    col_rotation=22.5):
    """
    Forked from https://github.com/albermax/innvestigate/blob/master/examples/utils.py
    """
    n_rows = len(grid)
    n_cols = len(grid[0])

    plt.clf()
    plt.rc("font", family="sans-serif", size=fontsize)

    f = plt.figure(figsize = (c*n_cols, c*n_rows))
    for r in range(n_rows):
        for c in range(n_cols):
            ax = plt.subplot2grid(shape=[n_rows, n_cols], loc=[r,c])
            ax.imshow(grid[r][c], interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])

            if not r: #column labels
                if col_labels != []:
                    ax.set_title(col_labels[c],
                                 rotation=col_rotation,
                                 horizontalalignment='left',
                                 verticalalignment='bottom')

                if super_col_labels != [] and c % 2 == 0:
                    label = super_col_labels[c // 2]
                    x_adjust = len(label) / 100
                    ax.text(x = 1 - x_adjust,
                            y = 1.2,
                            s = label,
                            transform=ax.transAxes,
                            fontdict={"fontsize": 50, "weight": 10})
            if not c: #row labels
                if row_labels_left != []:
                    txt_left = [l+'\n' for l in row_labels_left[r]]
                    ax.set_ylabel(''.join(txt_left),
                                  rotation=0,
                                  verticalalignment='center',
                                  horizontalalignment='right',
                                  )
            if c == n_cols-1:
                if row_labels_right != []:
                    txt_right = [l+'\n' for l in row_labels_right[r]]
                    ax2 = ax.twinx()
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    ax2.set_ylabel(''.join(txt_right),
                                  rotation=0,
                                  verticalalignment='center',
                                  horizontalalignment='left'
                                   )
    if not file_name:
        plt.show()
    else:
        print ('saving figure to {}'.format(file_name))
        plt.savefig(file_name, orientation='landscape', dpi=dpi, bbox_inches='tight')
