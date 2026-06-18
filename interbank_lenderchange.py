#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Generates a simulation of an interbank network
# Usage: interbank.py --help
#
#
# author: hector@bith.net
# date:   04/2023, 09/2025, 03/2026
import random
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import json
import warnings



class GraphStatistics:
    @staticmethod
    def giant_component_size(graph):
        """weakly connected componentes of the directed graph using Tarjan's algorithm:
           https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm"""
        if graph.is_directed():
            return nx.average_clustering(graph)
        else:
            return len(max(nx.connected_components(graph), key=len))

    @staticmethod
    def get_all_credit_channels(graph):
        return graph.number_of_edges()

    @staticmethod
    def avg_clustering_coef(graph):
        """clustering coefficient 0..1, 1 for totally connected graphs, and 0 for totally isolated
           if ~0 then a small world"""
        try:
            return nx.average_clustering(graph, count_zeros=True)
        except ZeroDivisionError:
            return 0

    @staticmethod
    def communities(graph):
        """Communities using greedy modularity maximization"""
        return list(nx.weakly_connected_components(graph if graph.is_directed() else graph.to_directed()))

    @staticmethod
    def grade_avg(graph):
        communities = GraphStatistics.communities(graph)
        total = 0
        for community in communities:
            total += len(community)
        return total / len(communities)

    @staticmethod
    def communities_not_alone(graph):
        """Number of communities that are not formed only with an isolated node"""
        total = 0
        for community in GraphStatistics.communities(graph):
            total += len(community) > 1
        return total

    @staticmethod
    def load_graph_json(filename):
        with open(filename, 'r') as f:
            graph_json = json.load(f)
            return nx.node_link_graph(graph_json, edges="links")

    @staticmethod
    def describe(graph):
        if isinstance(graph, str):
            graph_name = graph
            try:
                graph = GraphStatistics.load_graph_json(graph_name)
            except FileNotFoundError:
                print("json file does not exist: %s" % graph)
                sys.exit(0)
            except (UnicodeDecodeError, json.decoder.JSONDecodeError) as e:
                print("json file does not contain a valid graph: %s" % graph)
                sys.exit(0)
        else:
            graph_name = '?'
        communities = GraphStatistics.communities(graph)
        string_result = f"giant={GraphStatistics.giant_component_size(graph)} " + \
                        f" comm_not_alone={GraphStatistics.communities_not_alone(graph)}" + \
                        f" comm={len(communities)}" + \
                        f" gcs={GraphStatistics.giant_component_size(graph)}"
        return string_result


class LenderChange:
    SAVE_THE_DIFFERENT_GRAPH_OF_EACH_STEP = True
    GRAPH_NAME = "erdos_renyi"
    draw_guru = False
    draw_in_circle = True
    
    def __init__(self, model):
        self.p = model.config.p
        self.model = model
        self.node_positions = None
        self.node_colors = None

    def __len_edges(self, graph, node):
        if hasattr(graph, "in_edges"):
            return len(graph.in_edges(node))
        else:
            return len(graph.edges(node))

    def find_guru(self, graph):
        """It returns the guru ID and also a color_map with red for the guru, lightblue if weight<max/2 and blue others """
        guru_node = None
        guru_node_edges = 0
        for node in graph.nodes():
            edges_node = self.__len_edges(graph, node)
            if guru_node_edges < edges_node:
                guru_node_edges = edges_node
                guru_node = node
        return guru_node, guru_node_edges
    
    def __str__(self):
        return f"erdos_renyi p={self.p}"

    def __get_graph_from_guru(self, input_graph, output_graph, current_node, previous_node):
        """ It generates a new graph starting from the guru"""
        if self.__len_edges(input_graph, current_node) > 1:
            for (_, destination) in input_graph.edges(current_node):
                if destination != previous_node:
                    self.__get_graph_from_guru(input_graph, output_graph, destination, current_node)
        if previous_node is not None:
            output_graph.add_edge(current_node, previous_node)
            
    def get_graph_from_guru(self, input_graph):
        guru, _ = self.find_guru(input_graph)
        output_graph = nx.DiGraph()
        self.__get_graph_from_guru(input_graph, output_graph, guru, None)
        return output_graph, guru
    
    @staticmethod
    def _compute_dashed_edges(graph):
        edges = list(graph.edges())
        random.shuffle(edges)
        dashed = set()
        covered = set()
        target = len(graph.nodes()) // 2
        for u, v in edges:
            if u not in covered and v not in covered:
                dashed.add((u, v))
                covered.add(u)
                covered.add(v)
                if len(covered) >= target:
                    break
        return dashed

    @staticmethod
    def plot_saved_graph(json_filename):
        graph = GraphStatistics.load_graph_json(json_filename)
        output = json_filename.rsplit('.json', 1)[0] + 'b.png'
        lc = object.__new__(LenderChange)
        lc.node_positions = None
        lc.node_colors = None
        lc.model = None
        lc.draw(graph, new_guru_look_for=True, title=None, show=False, all_gray=True)
        plt.rcParams.update({'font.size': 6})
        plt.rcParams.update(plt.rcParamsDefault)
        warnings.filterwarnings("ignore", category=UserWarning)
        plt.savefig(output)
        plt.close('all')

    def draw(self, original_graph, new_guru_look_for=False, title=None, show=False, all_gray=False):
        """ Draws the graph using a spring layout that reuses the previous one layout, to show similar position for
            the same ids of nodes along time. If the graph is undirected (Barabasi) then no """
        graph_to_draw = original_graph.copy()
        plt.clf()
        if title:
            plt.title(title)
        # if not self.node_positions:
        guru = None
        if self.draw_in_circle:
            self.node_positions = nx.circular_layout(graph_to_draw)
        elif self.node_positions is None:
            self.node_positions = nx.spring_layout(graph_to_draw, pos=self.node_positions)
        if not hasattr(original_graph, "type") and original_graph.is_directed():
            # guru should not have out edges, and surely by random graphs it has:
            guru, _ = self.find_guru(graph_to_draw)
            for (i, j) in list(graph_to_draw.out_edges(guru)):
                graph_to_draw.remove_edge(i, j)
        if hasattr(original_graph, "type") and original_graph.type == "barabasi_albert":
            graph_to_draw, guru = self.get_graph_from_guru(graph_to_draw.to_undirected())
        if hasattr(original_graph, "type") and original_graph.type == "erdos_renyi" and graph_to_draw.is_directed():
            for node in list(graph_to_draw.nodes()):
                if not graph_to_draw.edges(node) and not graph_to_draw.in_edges(node):
                    graph_to_draw.remove_node(node)
            new_guru_look_for = True
        if not self.node_colors or new_guru_look_for:
            self.node_colors = []
            guru, guru_node_edges = self.find_guru(graph_to_draw)
            for node in graph_to_draw.nodes():
                if self.draw_guru and node == guru:
                    self.node_colors.append('darkorange')
                elif self.__len_edges(graph_to_draw, node) == 0:
                    self.node_colors.append('lightblue')
                elif self.__len_edges(graph_to_draw, node) == 1:
                    self.node_colors.append('steelblue')
                else:
                    self.node_colors.append('royalblue')
        dashed_edges = self._compute_dashed_edges(graph_to_draw) if not all_gray else set()
        nx.draw(graph_to_draw, pos=self.node_positions, node_color=self.node_colors,
                edge_color='lightgray', style='solid', arrows=False, with_labels=True)
        if dashed_edges:
            di_graph = nx.DiGraph()
            di_graph.add_nodes_from(graph_to_draw.nodes())
            di_graph.add_edges_from(dashed_edges)
            nx.draw_networkx_edges(di_graph, pos=self.node_positions,
                                  edgelist=list(dashed_edges), style='dashed',
                                  edge_color='black', arrows=True, arrowstyle='->')
        if show:
            plt.show()
        return guru

    def save_graph_png(self, graph, description, filename, add_info=False):
        if add_info:
            if not description:
                description = ""
            description += " " + GraphStatistics.describe(graph)

        guru = self.draw(graph, new_guru_look_for=True, title=description)
        plt.rcParams.update({'font.size': 6})
        plt.rcParams.update(plt.rcParamsDefault)
        warnings.filterwarnings("ignore", category=UserWarning)
        plt.savefig(filename)
        plt.close('all')
        return guru

    def save_graph_json(self, graph, filename):
        if graph:
            graph_json = nx.node_link_data(graph, edges="links")
            with open(filename, 'w') as f:
                json.dump(graph_json, f)

    def generate_banks_graph(self):
        result = nx.erdos_renyi_graph(n=self.model.config.N, p=self.model.config.p)
        return result, f"erdos_renyi p={self.model.config.p:5.3f} {GraphStatistics.describe(result)}"

    #TODO: quitar self.model
    # quitar export_datafile
    def setup_links(self, save_graph=True):
        """ It creates a Erdos Renyi graph with p defined in parameter['p']. No changes in relationships before end"""
        self.banks_graph, description = self.generate_banks_graph()
        self.banks_graph.type = self.GRAPH_NAME

        if save_graph and (save_graph == 'all' or self.model.t in save_graph):
             filename_for_file = f"_{self.GRAPH_NAME}"
             if self.SAVE_THE_DIFFERENT_GRAPH_OF_EACH_STEP:
                 filename_for_file += f"_{self.model.t}"
             export_datafile = self.model.stats.export_datafile
             if not export_datafile:
                 export_datafile = "results"
             destination_file_json = self.model.stats.get_export_path(export_datafile,
                                                                           f"{filename_for_file}.json")
             if not os.path.isfile(destination_file_json):
                 self.save_graph_json(self.banks_graph, destination_file_json)
                 self.save_graph_png(self.banks_graph, description, destination_file_json.replace('.json','.png'))

        self.model.log.debug("links ",
                             f"{self.GRAPH_NAME} (x,y)=x linked with y: " +
                             str(self.banks_graph.edges()) if self.banks_graph.edges() else "no edges in graph")
        for i in range(self.model.config.N):
            if self.model.d[i]>0:
                possible_lenders = []
                for (_, j) in self.banks_graph.edges(i):
                    if self.model.s[j]>0:
                        possible_lenders.append(j)
                if possible_lenders:
                    self.model.lenders[i] = random.choice(possible_lenders)
                else:
                    # no suitable lender possible for i:
                    self.model.lenders[i] = -1
            else:
                # i has money, it doesn't need a lender:
                self.model.lenders[i] = -1
        string_debug_lenders = ""
        for i in range(self.model.config.N):
            if self.model.lenders[i]>=0:
                string_debug_lenders += f"({i},{self.model.lenders[i]}) "
        self.model.log.debug("links ", "(x,y) x could borrows from y: "+
                             str(string_debug_lenders) if string_debug_lenders else f"no potential lenders/borrowers")
        return self.banks_graph

    def determine_current_communities(self):
        return len(GraphStatistics.communities(self.banks_graph))

    def determine_current_communities_not_alone(self):
        return GraphStatistics.communities_not_alone(self.banks_graph)

    def determine_current_graph_gcs(self):
        return GraphStatistics.giant_component_size(self.banks_graph)

    def determine_current_graph_grade_avg(self):
        return GraphStatistics.grade_avg(self.banks_graph)

