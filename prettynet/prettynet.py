# prettynet.py
#
# Bryan Daniels
# 2021/4/15
#
# Gathering together old code for drawing nice pictures of networks.
#

import networkx as nx
import tempfile
from IPython.display import Image
import os
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

def view(G, node_size=2000, font_size=22, font_color="white", figsize=(10,10), **kwargs):
    """
    Draws the graph G using a default networkx layout.
    
    For a description of keyword arguments, see documentation
    for prettynet.nx.draw.
    """
    plt.figure(figsize=figsize)
    
    # compute layout of nodes
    pos = nx.kamada_kawai_layout(G)
    
    # zoom out enough so nodes are not cut off
    # (why does matplotlib not do this already...?)
    pointunit = 200
    pad = np.sqrt(node_size)/pointunit
    xvals = [ point[0] for point in pos.values() ]
    yvals = [ point[1] for point in pos.values() ]
    xmin, xmax = min(xvals) - pad, max(xvals) + pad
    ymin, ymax = min(yvals) - pad, max(yvals) + pad
    plt.axis([xmin, xmax, ymin, ymax])
    
    nx.draw_networkx(G, pos,
                     node_size=node_size,
                     font_size=font_size,
                     font_color=font_color,
                     **kwargs)
    plt.axis('off')

def nodeColors(vals,nodeNames,cmap='PRGn'):
    """
    Converts a list of values to a list of colors for use with
    the `view` function, using the given colormap name.
    
    For other colormap names, see https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    # Rescale so all values are between zero and 1,
    # with zero in vals mapping to 0.5 in valsRescaled.
    valsRescaled = 0.5 * ( 1. + np.real_if_close(vals)/max(abs(vals)) )
    
    colormap = cm.get_cmap(cmap)
    colors = [ mpl.colors.to_hex(colormap(val)) for val in valsRescaled ]
    
    return colors

def nodeColorsDict(vals,nodeNames,cmap='PRGn'):
    """
    Converts a list of values to a nodecolors dictionary for use with
    view_jupyter_pygraphviz, using the given colormap name.
    
    For other colormap names, see https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    colors = nodeColors(vals,nodeNames,cmap=cmap)
    return dict(zip(nodeNames,colors))

def view_jupyter_pygraphviz(G, **kwargs):
    """
    Views the graph G in a jupyter notebook using pygraphviz.
    
    For a description of keyword arguments, see documentation
    for prettynet.view_pygraphviz.
    """
    
    path,graph = view_pygraphviz(G,show=False,**kwargs)
    
    return Image(G,filename=path)

# 4.24.2018 I copied this from networkx.nx_agraph
# in order to upgrade it (eventually submit as pull request to networkx?)
#
# 2021/8/12 We're not currently using this pygraphviz layout.
# If we do ever want to use it, we should probably rewrite all of this
# in more up-to-date networkx ways.  See
# https://networkx.org/documentation/stable/auto_examples/index.html#graphviz-layout
@nx.utils.open_file(6, 'w')
def view_pygraphviz(G, edgelabel=None, nodecolors=None, prog='dot', args='',
                       suffix='', path=None, fontcolors=None, sizes=None,
                       show=True):
    """Views the graph G using the specified layout algorithm.

    Parameters
    ----------
    G : NetworkX graph
        The machine to draw.
    edgelabel : str, callable, None
        If a string, then it specifes the edge attribute to be displayed
        on the edge labels. If a callable, then it is called for each
        edge and it should return the string to be displayed on the edges.
        The function signature of `edgelabel` should be edgelabel(data),
        where `data` is the edge attribute dictionary.
    prog : string
        Name of Graphviz layout program.
    args : str
        Additional arguments to pass to the Graphviz layout program.
    suffix : str
        If `filename` is None, we save to a temporary file.  The value of
        `suffix` will appear at the tail end of the temporary filename.
    path : str, None
        The filename used to save the image.  If None, save to a temporary
        file.  File formats are the same as those from pygraphviz.agraph.draw.

    Returns
    -------
    path : str
        The filename of the generated image.
    A : PyGraphviz graph
        The PyGraphviz graph instance used to generate the image.

    Notes
    -----
    If this function is called in succession too quickly, sometimes the
    image is not displayed. So you might consider time.sleep(.5) between
    calls if you experience problems.

    """
    if not len(G):
        raise nx.NetworkXException("An empty graph cannot be drawn.")

    import pygraphviz

    # If we are providing default values for graphviz, these must be set
    # before any nodes or edges are added to the PyGraphviz graph object.
    # The reason for this is that default values only affect incoming objects.
    # If you change the default values after the objects have been added,
    # then they inherit no value and are set only if explicitly set.

    # to_agraph() uses these values.
    attrs = ['edge', 'node', 'graph']
    for attr in attrs:
        if attr not in G.graph:
            G.graph[attr] = {}

    # These are the default values.
    edge_attrs = {'fontsize': '10'}
    node_attrs = {'style': 'filled',
                  'fillcolor': '#0000FF40',
                  'height': '0.75',
                  'width': '0.75',
                  'shape': 'circle'}
    graph_attrs = {}

    def update_attrs(which, attrs):
        # Update graph attributes. Return list of those which were added.
        added = []
        for k,v in attrs.items():
            if k not in G.graph[which]:
                G.graph[which][k] = v
                added.append(k)

    def clean_attrs(which, added):
        # Remove added attributes
        for attr in added:
            del G.graph[which][attr]
        if not G.graph[which]:
            del G.graph[which]

    # Update all default values
    update_attrs('edge', edge_attrs)
    update_attrs('node', node_attrs)
    update_attrs('graph', graph_attrs)

    # Convert to agraph, so we inherit default values
    A = nx.nx_agraph.to_agraph(G)

    # Remove the default values we added to the original graph.
    clean_attrs('edge', edge_attrs)
    clean_attrs('node', node_attrs)
    clean_attrs('graph', graph_attrs)

    # If the user passed in an edgelabel, we update the labels for all edges.
    if edgelabel is not None:
        if not hasattr(edgelabel, '__call__'):
            def func(data):
                return ''.join(["  ", str(data[edgelabel]), "  "])
        else:
            func = edgelabel

        # update all the edge labels
        if G.is_multigraph():
            for u,v,key,data in G.edges_iter(keys=True, data=True):
                # PyGraphviz doesn't convert the key to a string. See #339
                edge = A.get_edge(u,v,str(key))
                edge.attr['label'] = str(func(data))
        else:
            for u,v,data in G.edges_iter(data=True):
                edge = A.get_edge(u,v)
                edge.attr['label'] = str(func(data))
                
    # BCD If the user passed in nodecolors, we update the colors for all nodes.
    if nodecolors is not None:
        # update all the node colors
        for nodename in G.nodes():
                node = A.get_node(nodename)
                node.attr['fillcolor'] = nodecolors[nodename]
                
    # BCD If the user passed in fontcolors, we update the font colors for all nodes.
    if fontcolors is not None:
        # update all the node colors
        for nodename in G.nodes():
                node = A.get_node(nodename)
                node.attr['fontcolor'] = fontcolors[nodename]
                
    # BCD If the user passed in sizes, we update the sizes for all nodes.
    if sizes is not None:
        # update all the node colors
        for nodename in G.nodes():
                node = A.get_node(nodename)
                # make circle of fixed size (does not adapt to length of label)
                node.attr['fixedsize'] = True
                node.attr['width'] = sizes[nodename]
                node.attr['height'] = sizes[nodename]

    if path is None:
        ext = 'png'
        if suffix:
            suffix = '_%s.%s' % (suffix, ext)
        else:
            suffix = '.%s' % (ext,)
        path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    else:
        # Assume the decorator worked and it is a file-object.
        pass

    
    #nx.nx_agraph.display_pygraphviz(A, path=path, prog=prog, args=args, show=show)
    
    # (from outdated nx.nx_agraph.display_pygraphviz)
    filename = path.name
    format = os.path.splitext(filename)[1].lower()[1:]
    if not format:
        # Let the draw() function use its default
        format = None

    # Save to a file.  We must close the file before viewing it.
    A.draw(path, format, prog, args)
    path.close()
    
    # Optionally display in the default viewer
    if show:
        nx.utils.default_opener(filename)

    return path.name, A

