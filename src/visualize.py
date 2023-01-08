import matplotlib.pyplot as plt


def save_plot(model_id, train_losses, test_losses, train_accuracies, test_accuracies, epochs):
	fig, (ax1, ax2) = plt.subplots(2,1)
	fig.suptitle('Loss and Classification Accuracy', x=0.13, y=0.95, ha='left')
	
	ax1.plot(epochs, train_losses, 'o-', label='train')
	ax1.plot(epochs, test_losses, '.-', label='test')
	ax1.set_ylabel('Loss')
	ax1.legend(
		bbox_to_anchor=(0.66, 1.15, 0.35, 0.102),
		loc='upper right',
		ncol=2,
		mode='expand'
	)

	ax2.plot(epochs, train_accuracies, '.-', label='train')
	ax2.plot(epochs, test_accuracies, '.-', label='test')
	ax2.set_ylabel('Classification accuracy (%)')
	ax2.set_xlabel('Epochs')

	plt.savefig(f'./models/figures/knot-id_{model_id:04}.png')