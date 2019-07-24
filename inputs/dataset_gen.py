import importlib


class dataset_generator:
    def __init__(self, dataset_name):
        self._input_module = importlib.import_module('.input_' + dataset_name, package='inputs')
        self._dataset_properties = self._input_module.PROPERTIES

    def get_class_num(self):
        return self._dataset_properties['num_classes']

    def get_image_num(self):
        return self._dataset_properties['num_images']['train'], self._dataset_properties['num_images']['val']

    def input_fn(self, is_training, image_size, batch_size, num_epochs):
        dataset = self._input_module.input_fn(is_training, image_size, batch_size, num_epochs)

        return dataset
        


if __name__ == '__main__':
    gen = dataset_generator('lfw_id')

    pass
