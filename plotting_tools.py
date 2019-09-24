import networkx as nx
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def generate_random_3Dgraph(_Cells, zRatio, dCells, scale, seed=None):

    # Generate a dict of positions
    pos = {int(i): (scale*int(_Cells[i]['x']), scale*int(_Cells[i]['y']), scale*int(_Cells[i]['z'])/zRatio) for i in _Cells.keys()}

    # Create random 3D network
    G = nx.random_geometric_graph(len(_Cells), dCells, pos=pos)

    return G

def network_plot_3D(G, angle, save=False):

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()

    # Get the maximum number of edges adjacent to a single node
    edge_max = max([G.degree(i) for i in range(n)])

    # Define color range proportional to number of edges adjacent to a single node
    colors = [G.degree(i) for i in range(n)]
    cm = plt.cm.plasma

    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure(figsize=(10,8))
        ax = Axes3D(fig)



        # Loop on the pos dictionary to extract the x,y,z coordinates of each node

        for i,j in enumerate(G.edges()):

            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

        # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.2)

        x = []
        y = []
        z = []
        nodeColor = []
        s = []

        for key, value in pos.items():
            x.append(value[0])
            y.append(value[1])
            z.append(value[2])
            nodeColor.append(colors[key])
            s.append(20+20*G.degree(key))

        # Scatter plot
        sc = ax.scatter(x, y, z, c=nodeColor, cmap=cm, s=s, edgecolors='k', alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted


    # Set the initial view
    ax.view_init(30, angle)
    fig.patch.set_facecolor((1.0, 1, 1))
    ax.set_facecolor((1.0, 1, 1))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plt.rc('grid', linestyle="-", color='black')

    ax.set_xlabel('X axis ($\mu m$)')
    ax.set_ylabel('Y axis ($\mu m$)')
    ax.set_zlabel('Z axis ($\mu m$)')
    #ax.set_xlim(100, 250)
    #ax.set_ylim(100, 250)
    #ax.set_zlim(50, 90)


    # Hide the axes
    #ax.set_axis_off()

    #legend
    axins = inset_axes(ax,
                   width="2%",  # width = 5% of parent_bbox width
                   height="50%",  # height : 50%
                   loc='upper right',
                    bbox_to_anchor=(0., 0., 0.95, 0.8),
                   bbox_transform=ax.transAxes,
                   borderpad=0,
                   )
    cbar = fig.colorbar(sc, cax=axins)
    cbar.set_label('Node degree', rotation=270)
    cbar.ax.get_yaxis().labelpad = 15

    return

def classifier_plot_3D(G, angle, dic, save=False):

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()

    # Get the maximum number of edges adjacent to a single node
    edge_max = max([G.degree(i) for i in range(n)])

    # Define color range proportional to number of edges adjacent to a single node
    colors = []

    for key in dic['cells']:

        state = dic['cells'][key]['state']

        if state == 'Orange':

            colors.append('r')

        else: colors.append('b')

    cm = plt.cm.plasma

    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure(figsize=(10,8))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node

        for i,j in enumerate(G.edges()):

            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

        # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.2)

        x = []
        y = []
        z = []
        nodeColor = []
        s = []

        for key, value in pos.items():
            x.append(value[0])
            y.append(value[1])
            z.append(value[2])
            nodeColor.append(colors[key])
            s.append(20+20*G.degree(key))

        # Scatter plot
        sc = ax.scatter(x, y, z, c=nodeColor, cmap=cm, s=s, edgecolors='k', alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted


    # Set the initial view
    ax.view_init(30, angle)
    fig.patch.set_facecolor((1.0, 1, 1))
    ax.set_facecolor((1.0, 1, 1))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plt.rc('grid', linestyle="-", color='black')

    ax.set_xlabel('X axis ($\mu m$)')
    ax.set_ylabel('Y axis ($\mu m$)')
    ax.set_zlabel('Z axis ($\mu m$)')
    #ax.set_xlim(100, 250)
    #ax.set_ylim(100, 250)
    #ax.set_zlim(50, 90)


    # Hide the axes
    #ax.set_axis_off()

    return
