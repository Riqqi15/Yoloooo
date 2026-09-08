# Labelme JSON Drop Folder

Save human-drawn Labelme JSON files here. Each `imagePath` basename must match one image in `data/samples/`.

Accepted label: `tactile_paving`

Accepted shape type: `polygon`

An annotation containing zero shapes marks that image as reviewed negative. Prediction masks from `runs/sample-evaluation/tactile_masks/` must not be copied here or used as ground truth.
