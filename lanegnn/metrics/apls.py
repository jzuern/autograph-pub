
import networkx as nx
import scipy.stats
import numpy as np
from shapely.geometry import Point, LineString
import utm
import copy
import shapely
import math


###############################################################################
def insert_control_points(G_, control_points, max_distance_meters=10,
                          allow_renaming=True,
                          n_nodes_for_kd=1000, n_neighbors=20,
                          x_coord='x', y_coord='y',
                          verbose=True, super_verbose=False):
    """
    Wrapper around insert_point_into_G() for all control_points.
    Notes
    -----
    control_points are assumed to be of the format:
        [[node_id, x, y], ... ]
    TODO : Implement a version without renaming that tracks which node is
        closest to the desired point.
    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    control_points : array
        Points to insert in the graph, assumed to the of the format:
            [[node_id, x, y], ... ]
    max_distance_meters : float
        Maximum distance in meters between point and graph. Defaults to ``5``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    n_nodes_for_kd : int
        Minumu size of graph to render to kdtree to speed node placement.
        Defaults to ``1000``.
    n_neighbors : int
        Number of neigbors to return if building a kdtree. Defaults to ``20``.
    x_coord : str
        Name of x_coordinate, can be 'x' or 'lon'. Defaults to ``'x'``.
    y_coord : str
        Name of y_coordinate, can be 'y' or 'lat'. Defaults to ``'y'``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.
    Returns
    -------
    Gout, new_xs, new_ys : tuple
        Gout is the updated graph
        new_xs, new_ys are coordinates of the inserted points
    """


    # insertion can be super slow so construct kdtree if a large graph
    if len(G_.nodes()) > n_nodes_for_kd:
        # construct kdtree of ground truth
        kd_idx_dic, kdtree, pos_arr = apls_utils.G_to_kdtree(G_)
    # print("kd_idx_dic:", kd_idx_dic)
    # print("kdtree:", kdtree)
    # print("pos_arr:", pos_arr)

    Gout = G_.copy()
    new_xs, new_ys = [], []
    if len(G_.nodes()) == 0:
        return Gout, new_xs, new_ys

    for i, [node_id, x, y] in enumerate(control_points):

        if math.isinf(x) or math.isinf(y):
            print("Infinity in coords!:", x, y)
            return

        point = Point(x, y)

        # if large graph, determine nearby nodes
        if len(G_.nodes()) > n_nodes_for_kd:
            # get closest nodes
            node_names, dists_m_refine = apls_utils.nodes_near_point(
                x, y, kdtree, kd_idx_dic, x_coord=x_coord, y_coord=y_coord,
                # radius_m=radius_m,
                n_neighbors=n_neighbors,
                verbose=False)
            nearby_nodes_set = set(node_names)
        else:
            nearby_nodes_set = set([])

        # insert point
        Gout, node_props, xnew, ynew = insert_point_into_G(
            Gout, point, node_id=node_id,
            max_distance_meters=max_distance_meters,
            nearby_nodes_set=nearby_nodes_set,
            allow_renaming=allow_renaming,
            verbose=super_verbose)
        # xnew = node_props['x']
        # ynew = node_props['y']
        if (x != 0) and (y != 0):
            new_xs.append(xnew)
            new_ys.append(ynew)

    return Gout, new_xs, new_ys

###############################################################################
def create_edge_linestrings(G_, remove_redundant=True, verbose=False):
    """
    Ensure all edges have the 'geometry' tag, use shapely linestrings.
    Notes
    -----
    If identical edges exist, remove extras.
    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that may or may not include 'geometry'.
    remove_redundant : boolean
        Switch to remove identical edges, if they exist.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    Returns
    -------
    G_ : networkx graph
        Updated graph with every edge containing the 'geometry' tag.
    """

    # clean out redundant edges with identical geometry
    edge_seen_set = set([])
    geom_seen = []
    bad_edges = []

    for i, (u, v, data) in enumerate(G_.edges(data=True)):
        # create linestring if no geometry reported
        if 'geometry' not in data:
            sourcex, sourcey = G_.nodes[u]['x'],  G_.nodes[u]['y']
            targetx, targety = G_.nodes[v]['x'],  G_.nodes[v]['y']
            line_geom = LineString([Point(sourcex, sourcey),
                                    Point(targetx, targety)])
            data['geometry'] = line_geom

            # get reversed line
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
            # G_.edges[u][v]['geometry'] = lstring
        else:
            # check which direction linestring is travelling (it may be going
            #   from v -> u, which means we need to reverse the linestring)
            #   otherwise new edge is tangled
            line_geom = data['geometry']
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # print "dist_to_u, dist_to_v:", dist_to_u, dist_to_v
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
            if dist_to_u > dist_to_v:
                # data['geometry'].coords = list(line_geom.coords)[::-1]
                data['geometry'] = line_geom_rev
            # else:
            #    continue

        # flag redundant edges
        if remove_redundant:
            if i == 0:
                edge_seen_set = set([(u, v)])
                edge_seen_set.add((v, u))
                geom_seen.append(line_geom)

            else:
                if ((u, v) in edge_seen_set) or ((v, u) in edge_seen_set):
                    # test if geoms have already been seen
                    for geom_seen_tmp in geom_seen:
                        if (line_geom == geom_seen_tmp) \
                                or (line_geom_rev == geom_seen_tmp):
                            bad_edges.append((u, v))  # , key))
                else:
                    edge_seen_set.add((u, v))
                    geom_seen.append(line_geom)
                    geom_seen.append(line_geom_rev)

    if remove_redundant:
        for (u, v) in bad_edges:
            if G_.has_edge(u, v):
                G_.remove_edge(u, v)  # , key)

    return G_




def cut_linestring(line, distance, verbose=False):
    """
    Cuts a shapely linestring at a specified distance from its starting point.
    Notes
    ----
    Return orignal linestring if distance <= 0 or greater than the length of
    the line.
    Reference:
        http://toblerity.org/shapely/manual.html#linear-referencing-methods
    Arguments
    ---------
    line : shapely linestring
        Input shapely linestring to cut.
    distanct : float
        Distance from start of line to cut it in two.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    Returns
    -------
    [line1, line2] : list
        Cut linestrings.  If distance <= 0 or greater than the length of
        the line, return input line.
    """

    if verbose:
        print("Cutting linestring at distance", distance, "...")
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]

    # iterate through coorda and check if interpolated point has been passed
    # already or not
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pdl = line.project(Point(p))
        if verbose:
            print(i, p, "line.project point:", pdl)
        if pdl == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pdl > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

    # if we've reached here then that means we've encountered a self-loop and
    # the interpolated point is between the final midpoint and the the original
    # node
    i = len(coords) - 1
    cp = line.interpolate(distance)
    return [
        LineString(coords[:i] + [(cp.x, cp.y)]),
        LineString([(cp.x, cp.y)] + coords[i:])]



###############################################################################
def get_closest_edge_from_G(G_, point, nearby_nodes_set=set([]),
                            verbose=False):
    """
    Return closest edge to point, and distance to said edge.
    Notes
    -----
    Just discovered a similar function:
        https://github.com/gboeing/osmnx/blob/master/osmnx/utils.py#L501
    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    point : shapely Point
        Shapely point containing (x, y) coordinates.
    nearby_nodes_set : set
        Set of possible edge endpoints to search.  If nearby_nodes_set is not
        empty, only edges with a node in this set will be checked (this can
        greatly speed compuation on large graphs).  If nearby_nodes_set is
        empty, check all possible edges in the graph.
        Defaults to ``set([])``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    Returns
    -------
    best_edge, min_dist, best_geom : tuple
        best_edge is the closest edge to the point
        min_dist is the distance to that edge
        best_geom is the geometry of the ege
    """

    # get distances from point to lines
    dist_list = []
    edge_list = []
    geom_list = []
    p = point  # Point(point_coords)

    # for i, (u, v, key, data) in enumerate(nx.selfloop_edges(G_, keys=True, data=True)):
    # for i, (u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
    for i, edge in enumerate(G_.edges()):

        u = edge[0]
        v = edge[1]
        data = G_.edges[edge]
        key = 'geometry'

        # skip if u,v not in nearby nodes
        if len(nearby_nodes_set) > 0:
            if (u not in nearby_nodes_set) and (v not in nearby_nodes_set):
                continue
        try:
            line = data['geometry']
        except KeyError:
            line = data['attr_dict']['geometry']
        geom_list.append(line)
        dist_list.append(p.distance(line))
        edge_list.append([u, v, key])

    # get closest edge
    min_idx = np.argmin(dist_list)
    min_dist = dist_list[min_idx]
    best_edge = edge_list[min_idx]
    best_geom = geom_list[min_idx]

    return best_edge, min_dist, best_geom




###############################################################################
def insert_point_into_G(G_, point, node_id=100000, max_distance_meters=5,
                        nearby_nodes_set=set([]), allow_renaming=True,
                        verbose=False, super_verbose=False):
    """
    Insert a new node in the graph closest to the given point.
    Notes
    -----
    If the point is too far from the graph, don't insert a node.
    Assume all edges have a linestring geometry
    http://toblerity.org/shapely/manual.html#object.simplify
    Sometimes the point to insert will have the same coordinates as an
    existing point.  If allow_renaming == True, relabel the existing node.
    convert linestring to multipoint?
     https://github.com/Toblerity/Shapely/issues/190
    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    point : shapely Point
        Shapely point containing (x, y) coordinates
    node_id : int
        Unique identifier of node to insert. Defaults to ``100000``.
    max_distance_meters : float
        Maximum distance in meters between point and graph. Defaults to ``5``.
    nearby_nodes_set : set
        Set of possible edge endpoints to search.  If nearby_nodes_set is not
        empty, only edges with a node in this set will be checked (this can
        greatly speed compuation on large graphs).  If nearby_nodes_set is
        empty, check all possible edges in the graph.
        Defaults to ``set([])``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.
    Returns
    -------
    G_, node_props, min_dist : tuple
        G_ is the updated graph
        node_props gives the properties of the inserted node
        min_dist is the distance from the point to the graph
    """

    # check if node_id already exists in G
    # if node_id in set(G_.nodes()):
    #    print ("node_id:", node_id, "already in G, cannot insert node!")
    #    return

    best_edge, min_dist, best_geom = get_closest_edge_from_G(
        G_, point, nearby_nodes_set=nearby_nodes_set,
        verbose=super_verbose)
    [u, v, key] = best_edge
    G_node_set = set(G_.nodes())

    if verbose:
        print("Inserting point:", node_id)
        print("best edge:", best_edge)
        print("  best edge dist:", min_dist)
        u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
        v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
        print("ploc:", (point.x, point.y))
        print("uloc:", u_loc)
        print("vloc:", v_loc)

    if min_dist > max_distance_meters:
        if verbose:
            print("min_dist > max_distance_meters, skipping...")
        return G_, {}, -1, -1

    else:
        # updated graph

        # skip if node exists already
        if node_id in G_node_set:
            if verbose:
                print("Node ID:", node_id, "already exists, skipping...")
            return G_, {}, -1, -1

        # G_.edges[best_edge[0]][best_edge[1]][0]['geometry']
        line_geom = best_geom

        # Length along line that is closest to the point
        line_proj = line_geom.project(point)

        # Now combine with interpolated point on line
        new_point = line_geom.interpolate(line_geom.project(point))
        x, y = new_point.x, new_point.y

        #################
        # create new node

        try:
            # first get zone, then convert to latlon
            _, _, zone_num, zone_letter = utm.from_latlon(G_.nodes[u]['lat'],
                                                          G_.nodes[u]['lon'])
            # convert utm to latlon
            lat, lon = utm.to_latlon(x, y, zone_num, zone_letter)
        except:
            lat, lon = y, x

        # set properties
        # props = G_.nodes[u]
        node_props = {'highway': 'insertQ',
                      'lat': lat,
                      'lon': lon,
                      'osmid': node_id,
                      'x': x,
                      'y': y}
        # add node
        G_.add_node(node_id, **node_props)

        # assign, then update edge props for new edge
        _, _, edge_props_new = copy.deepcopy(
            list(G_.edges([u, v], data=True))[0])
        # remove extraneous 0 key

        # print ("edge_props_new.keys():", edge_props_new)
        # if list(edge_props_new.keys()) == [0]:
        #    edge_props_new = edge_props_new[0]

        # cut line
        split_line = cut_linestring(line_geom, line_proj)
        # line1, line2, cp = cut_linestring(line_geom, line_proj)
        if split_line is None:
            print("Failure in cut_linestring()...")
            print("type(split_line):", type(split_line))
            print("split_line:", split_line)
            print("line_geom:", line_geom)
            print("line_geom.length:", line_geom.length)
            print("line_proj:", line_proj)
            print("min_dist:", min_dist)
            return G_, {}, 0, 0

        if verbose:
            print("split_line:", split_line)

        # if cp.is_empty:
        if len(split_line) == 1:
            if verbose:
                print("split line empty, min_dist:", min_dist)
            # get coincident node
            outnode = ''
            outnode_x, outnode_y = -1, -1
            x_p, y_p = new_point.x, new_point.y
            x_u, y_u = G_.nodes[u]['x'], G_.nodes[u]['y']
            x_v, y_v = G_.nodes[v]['x'], G_.nodes[v]['y']
            # if verbose:
            #    print "x_p, y_p:", x_p, y_p
            #    print "x_u, y_u:", x_u, y_u
            #    print "x_v, y_v:", x_v, y_v

            # sometimes it seems that the nodes aren't perfectly coincident,
            # so see if it's within a buffer
            buff = 0.05  # meters
            if (abs(x_p - x_u) <= buff) and (abs(y_p - y_u) <= buff):
                outnode = u
                outnode_x, outnode_y = x_u, y_u
            elif (abs(x_p - x_v) <= buff) and (abs(y_p - y_v) <= buff):
                outnode = v
                outnode_x, outnode_y = x_v, y_v
            # original method with exact matching
            # if (x_p == x_u) and (y_p == y_u):
            #    outnode = u
            #    outnode_x, outnode_y = x_u, y_u
            # elif (x_p == x_v) and (y_p == y_v):
            #    outnode = v
            #    outnode_x, outnode_y = x_v, y_v
            else:
                print("Error in determining node coincident with node: "
                      + str(node_id) + " along edge: " + str(best_edge))
                print("x_p, y_p:", x_p, y_p)
                print("x_u, y_u:", x_u, y_u)
                print("x_v, y_v:", x_v, y_v)
                # return
                return G_, {}, 0, 0

            # if the line cannot be split, that means that the new node
            # is coincident with an existing node.  Relabel, if desired
            if allow_renaming:
                node_props = G_.nodes[outnode]
                # A dictionary with the old labels as keys and new labels
                #  as values. A partial mapping is allowed.
                mapping = {outnode: node_id}
                Gout = nx.relabel_nodes(G_, mapping)
                if verbose:
                    print("Swapping out node ids:", mapping)
                return Gout, node_props, x_p, y_p

            else:
                # new node is already added, presumably at the exact location
                # of an existing node.  So just remove the best edge and make
                # an edge from new node to existing node, length should be 0.0

                line1 = LineString([new_point, Point(outnode_x, outnode_y)])
                edge_props_line1 = edge_props_new.copy()
                edge_props_line1['length'] = line1.length
                edge_props_line1['geometry'] = line1
                # make sure length is zero
                if line1.length > buff:
                    print("Nodes should be coincident and length 0!")
                    print("  line1.length:", line1.length)
                    print("  x_u, y_u :", x_u, y_u)
                    print("  x_v, y_v :", x_v, y_v)
                    print("  x_p, y_p :", x_p, y_p)
                    print("  new_point:", new_point)
                    print("  Point(outnode_x, outnode_y):",
                          Point(outnode_x, outnode_y))
                    return

                # add edge of length 0 from new node to neareest existing node
                G_.add_edge(node_id, outnode, **edge_props_line1)
                return G_, node_props, x, y

                # originally, if not renaming nodes,
                # just ignore this complication and return the orignal
                # return G_, node_props, 0, 0

        else:
            # else, create new edges
            line1, line2 = split_line

            # get distances
            # print ("insert_point(), G_.nodes[v]:", G_.nodes[v])
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            # compare to first point in linestring
            geom_p0 = list(line_geom.coords)[0]
            # or compare to inserted point? [this might fail if line is very
            #    curved!]
            # geom_p0 = (x,y)
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # reverse edge order if v closer than u
            if dist_to_v < dist_to_u:
                line2, line1 = split_line

            if verbose:
                print("Creating two edges from split...")
                print("   original_length:", line_geom.length)
                print("   line1_length:", line1.length)
                print("   line2_length:", line2.length)
                print("   u, dist_u_to_point:", u, dist_to_u)
                print("   v, dist_v_to_point:", v, dist_to_v)
                print("   min_dist:", min_dist)

            # add new edges
            edge_props_line1 = edge_props_new.copy()
            edge_props_line1['length'] = line1.length
            edge_props_line1['geometry'] = line1
            # remove geometry?
            # edge_props_line1.pop('geometry', None)
            # line2
            edge_props_line2 = edge_props_new.copy()
            edge_props_line2['length'] = line2.length
            edge_props_line2['geometry'] = line2
            # remove geometry?
            # edge_props_line1.pop('geometry', None)

            # insert edge regardless of direction
            # G_.add_edge(u, node_id, **edge_props_line1)
            # G_.add_edge(node_id, v, **edge_props_line2)

            # check which direction linestring is travelling (it may be going
            # from v -> u, which means we need to reverse the linestring)
            # otherwise new edge is tangled
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # if verbose:
            #    print "dist_to_u, dist_to_v:", dist_to_u, dist_to_v
            if dist_to_u < dist_to_v:
                G_.add_edge(u, node_id, **edge_props_line1)
                G_.add_edge(node_id, v, **edge_props_line2)
            else:
                G_.add_edge(node_id, u, **edge_props_line1)
                G_.add_edge(v, node_id, **edge_props_line2)

            if verbose:
                print("insert edges:", u, '-', node_id, 'and', node_id, '-', v)

            # remove initial edge
            # G_.remove_edge(u, v, key)
            G_.remove_edge(u, v)

            return G_, node_props, x, y




def single_path_metric(len_gt, len_prop, diff_max=1):
    if len_gt <= 0:
        return 0
    elif len_prop < 0 and len_gt > 0:
        return diff_max
    else:
        diff_raw = np.abs(len_gt - len_prop) / len_gt
        return np.min([diff_max, diff_raw])


def path_sim_metric(all_pairs_lengths_gt, all_pairs_lengths_prop,
                    control_nodes=[], min_path_length=10,
                    diff_max=1, missing_path_len=-1, normalize=True,
                    verbose=False):
    """
    Compute metric for multiple paths.
    Notes
    -----
    Assume nodes in ground truth and proposed graph have the same names.
    Assume graph is undirected so don't evaluate routes in both directions
    control_nodes is the list of nodes to actually evaluate; if empty do all
        in all_pairs_lenghts_gt
    min_path_length is the minimum path length to evaluate
    https://networkx.github.io/documentation/networkx-2.2/reference/algorithms/shortest_paths.html
    Parameters
    ----------
    all_pairs_lengths_gt : dict
        Dictionary of path lengths for ground truth graph.
    all_pairs_lengths_prop : dict
        Dictionary of path lengths for proposal graph.
    control_nodes : list
        List of control nodes to evaluate.
    min_path_length : float
        Minimum path length to evaluate.
    diff_max : float
        Maximum value to return. Defaults to ``1``.
    missing_path_len : float
        Value to assign a missing path.  Defaults to ``-1``.
    normalize : boolean
        Switch to normalize outputs. Defaults to ``True``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    Returns
    -------
    C, diffs, routes, diff_dic
        C is the APLS score
        diffs is a list of the the route differences
        routes is a list of routes
        diff_dic is a dictionary of path differences
    """

    diffs = []
    routes = []
    diff_dic = {}
    gt_start_nodes_set = set(all_pairs_lengths_gt.keys())
    prop_start_nodes_set = set(all_pairs_lengths_prop.keys())

    if len(gt_start_nodes_set) == 0:
        return 0, [], [], {}

    # set nodes to inspect
    if len(control_nodes) == 0:
        good_nodes = list(all_pairs_lengths_gt.keys())
    else:
        good_nodes = control_nodes

    # iterate overall start nodes
    # for start_node, paths in all_pairs_lengths.iteritems():
    for start_node in good_nodes:
        if verbose:
            print("start node:", start_node)
        node_dic_tmp = {}

        # if we are not careful with control nodes, it's possible that the
        # start node will not be in all_pairs_lengths_gt, in this case use max
        # diff for all routes to that node
        # if the start node is missing from proposal, use maximum diff for
        # all possible routes to that node
        if start_node not in gt_start_nodes_set:
            print("for ss, node", start_node, "not in set")
            print("   skipping N paths:", len(
                list(all_pairs_lengths_prop[start_node].keys())))
            for end_node, len_prop in all_pairs_lengths_prop[start_node].items():
                diffs.append(diff_max)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff_max
            return

        paths = all_pairs_lengths_gt[start_node]

        # CASE 1
        # if the start node is missing from proposal, use maximum diff for
        # all possible routes to the start node
        if start_node not in prop_start_nodes_set:
            for end_node, len_gt in paths.items():
                if (end_node != start_node) and (end_node in good_nodes):
                    diffs.append(diff_max)
                    routes.append([start_node, end_node])
                    node_dic_tmp[end_node] = diff_max
            diff_dic[start_node] = node_dic_tmp
            # print ("start_node missing:", start_node)
            continue

        # else get proposed paths
        else:
            paths_prop = all_pairs_lengths_prop[start_node]

            # get set of all nodes in paths_prop, and missing_nodes
            end_nodes_gt_set = set(paths.keys()).intersection(good_nodes)
            # end_nodes_gt_set = set(paths.keys()) # old version with all nodes

            end_nodes_prop_set = set(paths_prop.keys())
            missing_nodes = end_nodes_gt_set - end_nodes_prop_set
            if verbose:
                print("missing nodes:", missing_nodes)

            # iterate over all paths from node
            for end_node in end_nodes_gt_set:
                # for end_node, len_gt in paths.iteritems():

                len_gt = paths[end_node]
                # skip if too short
                if len_gt < min_path_length:
                    continue

                # get proposed path
                if end_node in end_nodes_prop_set:
                    # CASE 2, end_node in both paths and paths_prop, so
                    # valid path exists
                    len_prop = paths_prop[end_node]
                else:
                    # CASE 3: end_node in paths but not paths_prop, so assign
                    # length as diff_max
                    len_prop = missing_path_len

                if verbose:
                    print("end_node:", end_node)
                    print("   len_gt:", len_gt)
                    print("   len_prop:", len_prop)

                # compute path difference metric
                diff = single_path_metric(len_gt, len_prop, diff_max=diff_max)
                diffs.append(diff)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff

            diff_dic[start_node] = node_dic_tmp

    if len(diffs) == 0:
        return 0, [], [], {}

    # compute Cost
    diff_tot = np.sum(diffs)
    if normalize:
        norm = len(diffs)
        diff_norm = diff_tot / norm
        C = 1. - diff_norm
    else:
        C = diff_tot

    return C, diffs, routes, diff_dic



###############################################################################
def create_graph_midpoints(G_, linestring_delta=50, is_curved_eps=0.03,
                           n_id_add_val=1, allow_renaming=True,
                           figsize=(0, 0),
                           verbose=False, super_verbose=False):
    """
    Insert midpoint nodes into long edges on the graph.
    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    linestring_delta : float
        Distance in meters between linestring midpoints. Defaults to ``50``.
    is_curved_eps : float
        Minumum curvature for injecting nodes (if curvature is less than this
        value, no midpoints will be injected). If < 0, always inject points
        on line, regardless of curvature.  Defaults to ``0.3``.
    n_id_add_val : int
        Sets min midpoint id above existing nodes
        e.g.: G.nodes() = [1,2,4], if n_id_add_val = 5, midpoints will
        be [9,10,11,...]
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    figsize : tuple
        Figure size for optional plot. Defaults to ``(0,0)`` (no plot).
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.
    Returns
    -------
    Gout, xms, yms : tuple
        Gout is the updated graph
        xms, yms are coordinates of the inserted points
    """

    # midpoint_loc = 0.5        # take the central midpoint for straight lines
    if len(G_.nodes()) == 0:
        return G_, [], []

    # midpoints
    xms, yms = [], []
    Gout = G_.copy()
    # midpoint_name_val, midpoint_name_inc = 0.01, 0.01
    midpoint_name_val, midpoint_name_inc = np.max(G_.nodes())+n_id_add_val, 1
    # for u, v, key, data in G_.edges(keys=True, data=True):
    for u, v, data in G_.edges(data=True):

        # curved line
        if 'geometry' in data:

            # first edge props and  get utm zone and letter
            edge_props_init = G_.edges([u, v])

            linelen = data['geometry'].length
            line = data['geometry']

            xs, ys = line.xy  # for plotting

            #################
            # check if curved or not
            minx, miny, maxx, maxy = line.bounds
            # get euclidean distance
            dst = scipy.spatial.distance.euclidean([minx, miny], [maxx, maxy])
            # ignore if almost straight
            if np.abs(dst - linelen) / linelen < is_curved_eps:
                # print "Line straight, skipping..."
                continue
            #################

            #################
            # also ignore super short lines
            if linelen < 0.75*linestring_delta:
                # print "Line too short, skipping..."
                continue
            #################

            if verbose:
                print("create_graph_midpoints()...")
                print("  u,v:", u, v)
                print("  data:", data)
                print("  edge_props_init:", edge_props_init)

            # interpolate midpoints
            # if edge is short, use midpoint, else get evenly spaced points
            if linelen <= linestring_delta:
                interp_dists = [0.5 * line.length]
            else:
                # get evenly spaced points
                npoints = len(np.arange(0, linelen, linestring_delta)) + 1
                interp_dists = np.linspace(0, linelen, npoints)[1:-1]
                if verbose:
                    print("  interp_dists:", interp_dists)

            # create nodes
            node_id_new_list = []
            xms_tmp, yms_tmp = [], []
            for j, d in enumerate(interp_dists):
                if verbose:
                    print("    ", j, "interp_dist:", d)

                midPoint = line.interpolate(d)
                xm0, ym0 = midPoint.xy
                xm = xm0[-1]
                ym = ym0[-1]
                point = Point(xm, ym)
                xms.append(xm)
                yms.append(ym)
                xms_tmp.append(xm)
                yms_tmp.append(ym)
                if verbose:
                    print("    midpoint:", xm, ym)

                # add node to graph, with properties of u
                node_id = midpoint_name_val
                # node_id = np.round(u + midpoint_name_val,2)
                midpoint_name_val += midpoint_name_inc
                node_id_new_list.append(node_id)
                if verbose:
                    print("    node_id:", node_id)

                # if j > 3:
                #    continue

                # add to graph
                Gout, node_props, _, _ = insert_point_into_G(
                    Gout, point, node_id=node_id,
                    allow_renaming=allow_renaming,
                    verbose=super_verbose)

    return Gout, xms, yms



def compute_apls_metric(all_pairs_lengths_gt_native,
                        all_pairs_lengths_prop_native,
                        all_pairs_lengths_gt_prime,
                        all_pairs_lengths_prop_prime,
                        control_points_gt, control_points_prop,
                        res_dir='', min_path_length=10,
                        verbose=False, super_verbose=False):
    """
    Compute APLS metric and plot results (optional)
    Notes
    -----
    Computes APLS and creates plots in res_dir (if it is not empty)
    Arguments
    ---------
    all_pairs_lengths_gt_native : dict
        Dict of paths for gt graph.
    all_pairs_lengths_prop_native : dict
        Dict of paths for prop graph.
    all_pairs_lengths_gt_prime : dict
        Dict of paths for gt graph with control points from prop.
    all_pairs_lengths_prop_prime : dict
        Dict of paths for prop graph with control points from gt.
    control_points_gt : list
        Array of control points.
    control_points_prop : list
        Array of control points.
    res_dir : str
        Output dir for plots.  Defaults to ``''`` (no plotting).
    min_path_length : float
        Minimum path length to evaluate.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.
    Returns
    -------
    C_tot, C_gt_onto_prop, C_prop_onto_gt : tuple
        C_tot is the total APLS score
        C_gt_onto_prop is the score when inserting gt control nodes onto prop
        C_prop_onto_gt is the score when inserting prop control nodes onto gt
    """

    # return 0 if no paths
    if (len(list(all_pairs_lengths_gt_native.keys())) == 0) \
            or (len(list(all_pairs_lengths_prop_native.keys())) == 0):
        return 0, 0, 0

    ####################
    # compute metric (gt to prop)

    control_nodes = [z[0] for z in control_points_gt]

    C_gt_onto_prop, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_gt_native,
        all_pairs_lengths_prop_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True,
        verbose=super_verbose)

    # compute metric (prop to gt)
    control_nodes = [z[0] for z in control_points_prop]


    C_prop_onto_gt, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_prop_native,
        all_pairs_lengths_gt_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True,
        verbose=super_verbose)

    # Total
    if (C_gt_onto_prop <= 0) or (C_prop_onto_gt <= 0) or (np.isnan(C_gt_onto_prop)) or (np.isnan(C_prop_onto_gt)):
        C_tot = 0
    else:
        C_tot = scipy.stats.hmean([C_gt_onto_prop, C_prop_onto_gt])
        if np.isnan(C_tot):
            C_tot = 0

    return C_tot, C_gt_onto_prop, C_prop_onto_gt


def add_travel_time(G_, speed_key='inferred_speed_mps', length_key='length',
                    travel_time_key='travel_time_s', default_speed=13.41,
                    verbose=False):
    """
    Compute and add travel time estimaes to each graph edge.
    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes speed.
    speed_key : str
        Key in the edge properties dictionary to use for the edge speed.
        Defaults to ``'inferred_speed_mps'``.
    length_key : str
        Key in the edge properties dictionary to use for the edge length.
        Defaults to ``'length'`` (asumed to be in meters).
    travel_time_key : str
        Name to assign travel time in the edge properties dictionary.
        Defaults to ``'travel_time_s'``.
    default_speed : float
        Default speed to use if speed_key is not found in edge properties
        Defaults to ``13.41`` (this is in m/s, and corresponds to 30 mph).
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    Returns
    -------
    G_ : networkx graph
        Updated graph with travel time attached to each edge.
    """

    for i, (u, v, data) in enumerate(G_.edges(data=True)):

        data[speed_key] = 1.

        if speed_key in data:
            speed = data[speed_key]
            if type(speed) == list:
                speed = np.mean(speed)
            # print("speed:", speed)
        else:
            print("speed_key not found:", speed_key)
            return
        if verbose:
            print("data[length_key]:", data[length_key])
            print("speed:", speed)
        travel_time_seconds = data[length_key] / speed
        data[travel_time_key] = travel_time_seconds

    return G_


def make_graphs(G_gt_, G_p_,
                weight='length',
                speed_key='inferred_speed_mps',
                travel_time_key='travel_time_s',
                max_nodes_for_midpoints=500,
                linestring_delta=50,
                is_curved_eps=0.012,
                max_snap_dist=4,
                allow_renaming=True,
                verbose=False,
                super_verbose=False):
    """
    Match nodes in ground truth and propsal graphs, and get paths.
    Notes
    -----
    The path length dictionaries returned by this function will be fed into
    compute_metric().
    Arguments
    ---------
    G_gt_ : networkx graph
        Ground truth graph.
    G_p_ : networkd graph
        Proposal graph over the same region.
    weight : str
        Key in the edge properties dictionary to use for the path length
        weight.  Defaults to ``'length'``.
    speed_key : str
        Key in the edge properties dictionary to use for the edge speed.
        Defaults to ``'inferred_speed_mps'``.
    travel_time_key : str
        Name to assign travel time in the edge properties dictionary.
        Defaults to ``'travel_time_s'``.
    max_nodes_for_midpoints : int
        Maximum number of gt nodes to inject midpoints.  If there are more
        gt nodes than this, skip midpoints and use this number of points
        to comput APLS.
    linestring_delta : float
        Distance in meters between linestring midpoints.
        If len gt nodes > max_nodes_for_midppoints this argument is ignored.
        Defaults to ``50``.
    is_curved_eps : float
        Minumum curvature for injecting nodes (if curvature is less than this
        value, no midpoints will be injected). If < 0, always inject points
        on line, regardless of curvature.
        If len gt nodes > max_nodes_for_midppoints this argument is ignored.
        Defaults to ``0.012``.
    max_snap_dist : float
        Maximum distance a node can be snapped onto a graph.
        Defaults to ``4``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.
    Return
    ------
    G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
            control_points_gt, control_points_prop, \
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime : tuple
        G_gt_cp  is ground truth with control points inserted
        G_p_cp is proposal with control points inserted
        G_gt_cp_prime is ground truth with control points from prop inserted
        G_p_cp_prime is proposal with control points from gt inserted
        all_pairs_lengths_gt_native is path length dict corresponding to G_gt_cp
        all_pairs_lengths_prop_native is path length dict corresponding to G_p_cp
        all_pairs_lengths_gt_prime is path length dict corresponding to G_gt_cp_prime
        all_pairs_lenfgths_prop_prime is path length dict corresponding to G_p_cp_prime
    """

    # for i, (u, v, data) in enumerate(nx.selfloop_edges(G_gt_, keys=False, data=True)):
    # for i, (u, v, data) in enumerate(G_gt_.edges(keys=False, data=True)):
    for i, edge in enumerate(G_gt_.edges()):

        u = edge[0]
        v = edge[1]
        data = G_gt_.edges[edge]

        if weight not in data.keys():
            print("Error!", weight, "not in G_gt_ edge u, v, data:", u, v, data)
            return

    # for i, (u, v, key, data) in enumerate(nx.selfloop_edges(G_gt_, keys=True, data=True)):
    # for i, (u, v, key, data) in enumerate(G_gt_.edges(keys=True, data=True)):
    for i, edge in enumerate(G_gt_.edges()):

        u = edge[0]
        v = edge[1]
        data = G_gt_.edges[edge]
        key = 'geometry'

        try:
            line = data['geometry']
        except KeyError:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    # create graph with midpoints
    G_gt0 = create_edge_linestrings(G_gt_.to_undirected())


    G_gt_cp0, xms, yms = create_graph_midpoints(
        G_gt0.copy(),
        linestring_delta=linestring_delta,
        figsize=(0, 0),
        is_curved_eps=is_curved_eps,
        verbose=False)
    # add travel

    G_gt_cp = add_travel_time(G_gt_cp0.copy(),
                              speed_key=speed_key,
                              travel_time_key=travel_time_key)

    # get ground truth control points
    control_points_gt = []
    for n in G_gt_cp.nodes():
        u_x, u_y = G_gt_cp.nodes[n]['x'], G_gt_cp.nodes[n]['y']
        control_points_gt.append([n, u_x, u_y])

    all_pairs_lengths_gt_native = dict(nx.shortest_path_length(G_gt_cp, weight=weight))

    # Proposal
    for i, edge in enumerate(G_p_.edges()):

        u = edge[0]
        v = edge[1]
        data = G_p_.edges[edge]

        if weight not in data.keys():
            print("Error!", weight, "not in G_p_ edge u, v, data:", u, v, data)
            return

    # get proposal graph with native midpoints
    for i, edge in enumerate(G_p_.edges()):

        u = edge[0]
        v = edge[1]
        data = G_p_.edges[edge]
        key = 'geometry'

        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    G_p0 = create_edge_linestrings(G_p_.to_undirected())

    # add travel time
    G_p = add_travel_time(G_p0.copy(),
                          speed_key=speed_key,
                          travel_time_key=travel_time_key)

    G_p_cp0, xms_p, yms_p = create_graph_midpoints(
        G_p.copy(),
        linestring_delta=linestring_delta,
        figsize=(0, 0),
        is_curved_eps=is_curved_eps,
        verbose=False)

    # add travel time
    G_p_cp = add_travel_time(G_p_cp0.copy(),
                             speed_key=speed_key,
                             travel_time_key=travel_time_key)

    # set proposal control nodes, originally just all nodes in G_p_cp
    # original method sets proposal control points as all nodes in G_p_cp
    # get proposal control points
    control_points_prop = []
    for n in G_p_cp.nodes():
        u_x, u_y = G_p_cp.nodes[n]['x'], G_p_cp.nodes[n]['y']
        control_points_prop.append([n, u_x, u_y])

    # get paths
    all_pairs_lengths_prop_native = dict(
        nx.shortest_path_length(G_p_cp, weight=weight))

    # insert gt control points into proposal
    G_p_cp_prime0, xn_p, yn_p = insert_control_points(
        G_p.copy(), control_points_gt,
        max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming,
        verbose=super_verbose)

    # add travel time
    G_p_cp_prime = add_travel_time(G_p_cp_prime0.copy(),
                                   speed_key=speed_key,
                                   travel_time_key=travel_time_key)

    # now insert control points into ground truth
    # permit renaming of inserted nodes if coincident with existing node
    G_gt_cp_prime0, xn_gt, yn_gt = insert_control_points(
        G_gt_,
        control_points_prop,
        max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming,
        verbose=super_verbose)

    # add travel time
    G_gt_cp_prime = add_travel_time(G_gt_cp_prime0.copy(),
                                    speed_key=speed_key,
                                    travel_time_key=travel_time_key)

    # get paths
    all_pairs_lengths_gt_prime = dict(
        nx.shortest_path_length(G_gt_cp_prime, weight=weight))
    all_pairs_lengths_prop_prime = dict(
        nx.shortest_path_length(G_p_cp_prime, weight=weight))


    return G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, control_points_gt, control_points_prop, \
        all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
        all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime


def to_nx(target, output_pairs, output):

    # make GT graph
    relation_labels = target.get_field("relation_labels").cpu().numpy()
    target_bbox = target.bbox.cpu().numpy()
    target_points = []

    for bbox in target_bbox:
        x = (bbox[0] + bbox[2] ) / 2.0
        y = (bbox[1] + bbox[3] ) / 2.0
        target_points.append(Point(x, y))

    G_gt_init = nx.Graph()

    for i, p in enumerate(target_points):
        G_gt_init.add_node(i, x=p.x, y=p.y)

    for d in relation_labels:
        p0 = target_points[d[0]]
        p1 = target_points[d[1]]
        line = LineString([p0, p1])
        G_gt_init.add_edge(d[0], d[1], geometry=line, length=line.length)

    # make predicted graph
    G_p_init = nx.Graph()

    relation_pred = output_pairs.get_field('idx_pairs').cpu().numpy()
    prediction_bbox = output.bbox.cpu().numpy()
    prediction_points = []

    for bbox in prediction_bbox:
        x = (bbox[0] + bbox[2] ) / 2.0
        y = (bbox[1] + bbox[3] ) / 2.0
        prediction_points.append(Point(x, y))

    for i, p in enumerate(prediction_points):
        G_p_init.add_node(i, x=p.x, y=p.y)

    for d in relation_pred:
        p0 = prediction_points[d[0]]
        p1 = prediction_points[d[1]]
        line = LineString([p0, p1])
        G_p_init.add_edge(d[0], d[1], geometry=line, length=line.length)

    return G_gt_init, G_p_init


def get_apls(target, output_pairs, output):

    # settings
    weight = 'length'
    speed_key = 'inferred_speed_mps'
    travel_time_key = 'travel_time_s'
    linestring_delta = 20.  # Distance between linestring midpoints.
    is_curved_eps = -1.  # can inject nodes everywhere
    max_snap_dist = 5.
    allow_renaming = 1

    G_gt_init, G_p_init = to_nx(target, output_pairs, output)


    G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, control_points_gt, control_points_prop, all_pairs_lengths_gt_native, \
    all_pairs_lengths_prop_native, all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime = \
        make_graphs(G_gt_init, G_p_init,
                      weight=weight,
                      speed_key=speed_key,
                      travel_time_key=travel_time_key,
                      linestring_delta=linestring_delta,
                      is_curved_eps=is_curved_eps,
                      max_snap_dist=max_snap_dist,
                      allow_renaming=allow_renaming,
                      verbose=False)

    apls_metric, apls_metric_gt_onto_prop, apls_metric_prop_onto_gt = compute_apls_metric(
        all_pairs_lengths_gt_native, all_pairs_lengths_prop_native,
        all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime,
        control_points_gt, control_points_prop,
        min_path_length=0.1,
        verbose=False, res_dir=None)

    return apls_metric


if __name__ == '__main__':

    get_apls(None, None)