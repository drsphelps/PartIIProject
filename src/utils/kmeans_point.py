class KMeansPoint():
    def __init__(self, coords, constraint=-1, text=""):
        self.original_text = text
        self.coords = coords
        self.constraint = constraint
        self.cluster = -1

    def setCluster(self, cluster):
        self.cluster = cluster
