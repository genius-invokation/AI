import logging
import os
from datetime import datetime

class MultiFileLogger(logging.Logger):
    def __init__(self, name, default_log_file=None, log_dir='logs', level=logging.INFO):
        super().__init__(name, level)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.formatter = logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.loggers = {}  # Store loggers for different files
        
        # Default file handler
        if default_log_file:
            self.add_file_handler(default_log_file)
            self.default_log_file = default_log_file
        else:
            self.default_log_file = None
    
    def add_file_handler(self, log_file):
        """Add a new file handler for a specific log file."""
        if log_file not in self.loggers:
            log_path = os.path.join(self.log_dir, log_file)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(self.formatter)
            
            # Create a logger for the specific file
            file_logger = logging.getLogger(f"{self.name}_{log_file}")
            file_logger.setLevel(self.level)
            file_logger.addHandler(file_handler)
            self.loggers[log_file] = file_logger

    def log_to_file(self, level, message, log_file=None):
        """Log a message to the specified log file or default."""
        if log_file:
            if log_file not in self.loggers:
                self.add_file_handler(log_file)
            # Redirect the log call to the specific logger
            self.loggers[log_file].log(level, message)
        else:
            # Use default logger if no log_file is provided
            if self.default_log_file:
                if self.default_log_file not in self.loggers:
                    self.add_file_handler(self.default_log_file)
                self.loggers[self.default_log_file].log(level, message)
            else:
                super().log(level, message)

    def info(self, message, *args, log_file=None, **kwargs):
        """Log an INFO message."""
        self.log_to_file(logging.INFO, message, log_file=log_file)

    def error(self, message, *args, log_file=None, **kwargs):
        """Log an ERROR message."""
        self.log_to_file(logging.ERROR, message, log_file=log_file)

# Factory function for creating the logger
def setup_logger(name, default_log_file=None, log_dir='logs', level=logging.INFO):
    logging.setLoggerClass(MultiFileLogger)
    logger = logging.getLogger(name)
    if not isinstance(logger, MultiFileLogger):
        MultiFileLogger(name, default_log_file, log_dir, level)
    return logger

# Example Usage
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
default_log_file = f'game_actions_{timestamp}.log'

# Setup the logger
logger = setup_logger('default', default_log_file)

# Log to specific files dynamically
# logger.info("Log entry to zzgame_start.log", log_file="zzgame_start.log")
# logger.error("Error logged to zzerrors.log", log_file="zzerrors.log")
