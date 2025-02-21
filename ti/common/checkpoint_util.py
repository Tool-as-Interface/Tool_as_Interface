from typing import Optional, Dict
import os

class TopKCheckpointManager:
    """
    A manager for saving top K performing model checkpoints based on a specified metric.

    Attributes:
        save_dir (str): Directory where checkpoints are saved.
        monitor_key (str): Key for the metric to monitor in the data dictionary.
        mode (str): Mode for evaluation ('max' for higher is better, 'min' for lower is better).
        k (int): Number of top-performing checkpoints to keep.
        format_str (str): Format string for generating checkpoint filenames.
        path_value_map (dict): A dictionary mapping checkpoint paths to their metric values.
    """

    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        """
        Initializes the TopKCheckpointManager with the specified configuration.

        Args:
            save_dir (str): The directory to save checkpoints.
            monitor_key (str): The metric key to monitor for checkpointing.
            mode (str, optional): Determines if 'max' or 'min' values of the monitored metric are better. Defaults to 'min'.
            k (int, optional): The number of top checkpoints to maintain. Defaults to 1.
            format_str (str, optional): Format string for checkpoint filenames. Defaults to 'epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'.
        """

        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        """
        Determines the checkpoint path for the current metrics and decides whether to replace an existing checkpoint.

        Args:
            data (Dict[str, float]): A dictionary containing current metric data.

        Returns:
            Optional[str]: The path to save the new checkpoint if it is among the top K; otherwise, None.
        """

        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path
