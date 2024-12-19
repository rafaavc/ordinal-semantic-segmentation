from matplotlib import pyplot as plt

def display_confusion_matrix(mx):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    mx = [ [ round(n, 4) for n in line ] for line in mx ]
    ax.matshow(mx, cmap=plt.cm.Reds, alpha=1)
    for i in range(len(mx)):
        for j in range(len(mx[0])):
            ax.text(x=j, y=i, s=mx[i][j], va='center', ha='center', size='xx-large')
    
    plt.ylabel('Predictions', fontsize=18)
    plt.xlabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
