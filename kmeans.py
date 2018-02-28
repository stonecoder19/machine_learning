

def scalar_multiply(c, v):
    return[c * v_i for v_i in v]

def vector_sum(v1, v2):
   return[ v1_i + v2_i for v1_i,v2_i in zip(v1,v2)]

def vector_mean(vectors):
    return scalar_multiply(1/n, vector_sum(vectors))

class KMeans:
   

    def __init__(self, k):
        self.k = k
        self.means = None

    def classify(self, input):
        return min(range(self.k),
                   key=lambda i: squared_distance(input,self.means[i]))

    def train(self, inputs):
        self.means = random.sample(inputs, self.k)
        assignments = None
   
        while True:
            new_assignments = map(self.classify, inputs)
          
            if assignments == new_assignments:
                return

            assignments = new_assignments

            for i in range(self.k):
                i_points = [p for p, a in zip(inputs, assignments) if a == i]
       
                if i_points:
                    self.means[i] = vector_mean(i_points)



def squared_clustering_errors(inputs, k):
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = map(clusterer.classify, inputs)

    return sum(squared_distance(input, means[cluster])
              for input, cluster in zip(inputs, assignments))


ks = range(1, len(inputs + 1)
errors = [squared_clustering_errors(inputs, k) for k in ks]

plt.plot(ks, errors)
plt.xticks(ks)
plt.xlabel("k")
plt.ylabel("total squared error")
plt.title("Total Error vs.  of Clusters")
plt.show()
