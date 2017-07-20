from scipy.io import arff
import pandas as pd
import math

# FileNames saved from weka after doing the clustering
fileName = ['data/cluster2.arff','data/cluster3.arff','data/cluster5.arff']
# Number of clusters we will be needing
numberOfClusters = [2, 3, 5]
# Starting iteration on filenames
for i in range(0, len(fileName)):
    # Loading arff file
    data, meta = arff.loadarff(fileName[i])
    # Making a dataframe
    dataset = pd.DataFrame(data)
    # Sorting value according to Cluster column
    dataset.sort_values('Cluster', inplace=True)
    # Resetting the indexes
    dataset.reset_index(inplace=True)
    # Counting number of instances in each cluster
    clusterCounts = dataset.drop_duplicates().groupby('Cluster').size()
    # Array containing s(i)
    s = []
    # Starting the loop for a(i) object
    for j in range(0, len(dataset)):
        jthSl      = dataset['sepallength'][j]
        jthSw      = dataset['sepalwidth'][j]
        jthPl      = dataset['petallength'][j]
        jthPw      = dataset['petalwidth'][j]
        jthCluster = dataset['Cluster'][j]
        b = {}
        a = 0
        sj = 0
        intraClustDist = 0
        bag = []
        # Starting for loop for b objects
        for k in range(0, len(dataset)):
            if k == j:
                continue
            kthSl = dataset['sepallength'][k]
            kthSw = dataset['sepalwidth'][k]
            kthPl = dataset['petallength'][k]
            kthPw = dataset['petalwidth'][k]
            # Calculating intra-cluster distance
            if jthCluster == dataset['Cluster'][k]:
                intraClustDist += math.sqrt( (kthSl - jthSl)**2 + (kthSw - jthSw)**2 + (kthPl - jthPl)**2 + (kthPw - jthPw)**2 )
            # Calculating inter-cluster distance
            else:
                key = dataset['Cluster'][k]
                # If the cluster is already there
                if key in b:
                    b[key] += math.sqrt( (kthSl - jthSl)**2 + (kthSw - jthSw)**2 + (kthPl - jthPl)**2 + (kthPw - jthPw)**2 )
                # If the cluster is not already there
                else:
                    b[key] = math.sqrt( (kthSl - jthSl)**2 + (kthSw - jthSw)**2 + (kthPl - jthPl)**2 + (kthPw - jthPw)**2 )
        # Get average value of b(i) for each cluster
        for k, v in b.iteritems():
            val = v
            N = clusterCounts[k]
            b = (val/N)
            bag.append(b)
        # Cluster count
        aN = clusterCounts[jthCluster]
        a  = intraClustDist
        # Get a(i)
        aj = (a/aN)
        # Get the nearest cluster distance with the object or b(i)
        bj = min(bag)
        # Get s(i)
        sj = (bj - aj)/max(aj, bj)
        s.append(sj)
    averageSilhouette = reduce(lambda x, y: x + y, s) / len(s)
    print "\n"
    print "Average Silhouette for K = ",numberOfClusters[i]
    print "Average Silhouette: ", averageSilhouette
    print "================================================="