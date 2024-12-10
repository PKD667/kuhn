import numpy as np

class Op():
    def __init__(self):
        self.add = 0
        self.mul = 0
    
    def apply(self, x):
        return (x  * self.mul) + self.add

    def set(self, add, mul):
        self.add = add
        self.mul = mul
    
    def update(self, dadd, dmul):
        self.add += dadd
        self.mul += dmul
        if self.mul < 0.1:
            self.mul = 0.1
        if self.mul > 10:
            self.mul = 10
        if self.add < -1:
            self.add = -1
        if self.add > 1:
            self.add = 1

class Cluster():

    def __init__(self):
        self.stimuli = []
        self.data = []
        self.mean = 0

    def add(self, stimuli, datum):
        self.stimuli.append(stimuli)
        self.data.append(datum)
        self.update()

    def update(self):
        self.mean = np.mean(self.data, axis=0)

    def distance(self, x):
        return np.linalg.norm(x - self.mean)

class Transform():
    transformations = []  
    clusters = []   
    dist_threshold = 0.1

    def __init__(self,n):
        # n is the number of input stimuli
        self.n = n
        for i in range(n):
            self.transformations.append(Op())

    def process(self, x):
        y = np.zeros(self.n)
        for i in range(self.n):
            y[i] = self.transformations[i].apply(x[i])
        return y

    def set(self, add, mul, i):
        self.transformations[i].set(add, mul)

    def clusterize(self,stimuli):
        
        data = []
        for i in range(len(stimuli)):
            d = self.apply(stimuli[i])
            print(f"stimuli: {stimuli[i]} -> {d}")

        # data is a list of vectors
        clusters = []
        for i in range(len(data)):
            for j in range(len(clusters)):
                print(f"norm: {np.linalg.norm(data[i] - clusters[j][0])}")
                if clusters[j].distance(data[i]) < self.dist_threshold:
                    clusters[j].add(stimuli[i], data[i])
                    break
            else:
                clusters.append(Cluster())
                clusters[-1].add(stimuli[i], data[i])
    
    def learn(self, label_clusters):
        
        # using the given stimuli and their label clusters, update the transformations    
        for i in range(len(label_clusters)):
            # try to update the transformations so each stimuli in the cluster is close ot the mean
            for j in range(len(label_clusters[i].stimuli)):
                stimuli = label_clusters[i].stimuli[j]
                datum = self.process(label_clusters[i].stimuli[j])
                while label_clusters[i].distance(datum) > self.dist_threshold:
                    print(f"{j} - distance: {label_clusters[i].distance(datum)}")
                    # randomize the transformations by a small amount
                    for k in range(self.n):
                        self.transformations[k].update(np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1))
                    datum = self.process(stimuli)


def create_cluster(n, mean, std, size):
    cluster = Cluster()
    cluster.data = np.random.randn(size, n) * std + mean
    # now create a transformation of size n that convets this data to seemingly random stimuli
    t = Transform(n)
    for i in range(n):
        t.set(np.random.uniform(-1, 1), np.random.uniform(-1, 1), i)
    for i in range(size):
        print(f"{cluster.data[i]} -> {t.process(cluster.data[i])}")
        cluster.stimuli.append(t.process(cluster.data[i]))
    return cluster

if __name__ == "__main__":
    
    n_clusters = 3
    n = 2

    label_clusters = []
    for i in range(n_clusters):
        label_clusters.append(create_cluster(n, np.random.uniform(-1, 1, n), np.random.uniform(0.1, 0.5), 10))
    
    t = Transform(n)
    t.learn(label_clusters)






    