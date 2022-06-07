def main(file_name, patience, test_patience_values):
    import argparse
    import os
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import ScalarFormatter

    # load file
    file = open(file_name, 'r')
    # find line starting with "Loss"
    loss_text = None
    for line in file:
        if line.startswith('Loss'):
            loss_text = line.split(':')[1].strip()
            # trim brackets
            loss_text = loss_text[1:-1]
    if loss_text is not None:
        # convert array text to array
        loss_array = np.array(loss_text.split(','), dtype=np.float32)
        print(f"Half way Loss: {loss_array[int(len(loss_array)/2)]}")
        # plot
        fig, ax = plt.subplots(figsize=(8, 6))
        # plotting only second half of loss array
        ax.plot(loss_array[len(loss_array)//2:], label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.xaxis.set_major_locator(MaxNLocator())
        ax.yaxis.set_major_locator(MaxNLocator())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        plt.show()

        # Simulate different patience values
        if test_patience_values is not None:
            test_patience_values.sort()
            best_loss = np.min(loss_array)
            print('Best loss: {}'.format(best_loss))
            # assert loss_array[-patience-1] == best_loss
            counter = 0
            best_loss = loss_array[0]+1
            for i in range(len(loss_array)):
                loss = loss_array[i]
                if loss >= best_loss:
                    counter += 1
                else:
                    counter = 0
                    best_loss = loss
                if counter >= test_patience_values[0]:
                    print(f"{i} epochs Patience {test_patience_values[0]} with best loss {best_loss}")
                    test_patience_values.pop(0)
                    if len(test_patience_values) == 0:
                        break



if __name__ == '__main__':
    main(
        r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220412 5000epochs for 100fps spf\report 20220412 123228\Report-spf-d1.txt"
    , 50, [1, 20, 30, 40, 50])
