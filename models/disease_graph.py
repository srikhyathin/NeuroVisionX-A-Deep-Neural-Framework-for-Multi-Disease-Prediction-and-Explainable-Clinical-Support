import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os


class DiseaseKnowledgeGraph:

    def __init__(self):

        self.graph = nx.DiGraph()

        self.description_file = "data/symptoms/symptom_Description.csv"
        self.precaution_file = "data/symptoms/symptom_precaution.csv"

        self.load_data()

    # ---------------------------------------------------

    def load_data(self):

        if os.path.exists(self.description_file):

            desc_df = pd.read_csv(self.description_file)

            for _, row in desc_df.iterrows():

                disease = row["Disease"]

                description = row["Description"]

                self.graph.add_node(
                    disease,
                    type="Disease"
                )

                self.graph.add_node(
                    description,
                    type="Description"
                )

                self.graph.add_edge(
                    disease,
                    description,
                    relation="has_description"
                )

        if os.path.exists(self.precaution_file):

            prec_df = pd.read_csv(self.precaution_file)

            for _, row in prec_df.iterrows():

                disease = row["Disease"]

                self.graph.add_node(
                    disease,
                    type="Disease"
                )

                for i in range(1,5):

                    precaution = row[f"Precaution_{i}"]

                    if pd.notna(precaution):

                        self.graph.add_node(
                            precaution,
                            type="Precaution"
                        )

                        self.graph.add_edge(
                            disease,
                            precaution,
                            relation="precaution"
                        )

    # ---------------------------------------------------

    def visualize(self):

        plt.figure(figsize=(16,12))

        pos = nx.spring_layout(
            self.graph,
            seed=42
        )

        nx.draw_networkx_nodes(

            self.graph,

            pos,

            node_color="skyblue",

            node_size=1200

        )

        nx.draw_networkx_edges(

            self.graph,

            pos,

            arrows=True

        )

        nx.draw_networkx_labels(

            self.graph,

            pos,

            font_size=8

        )

        plt.title("NeuroVisionX Disease Knowledge Graph")

        plt.axis("off")

        plt.show()

    # ---------------------------------------------------

    def get_description(self, disease):

        if disease not in self.graph:

            return None

        for node in self.graph.successors(disease):

            if self.graph[disease][node]["relation"] == "has_description":

                return node

        return None

    # ---------------------------------------------------

    def get_precautions(self, disease):

        precautions = []

        if disease not in self.graph:

            return precautions

        for node in self.graph.successors(disease):

            if self.graph[disease][node]["relation"] == "precaution":

                precautions.append(node)

        return precautions

    # ---------------------------------------------------

    def summary(self):

        print("="*50)

        print("NeuroVisionX Disease Knowledge Graph")

        print("="*50)

        print("Nodes :", self.graph.number_of_nodes())

        print("Edges :", self.graph.number_of_edges())

        print("="*50)