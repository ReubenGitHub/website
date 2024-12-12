import time
import threading

class ModelManager:
    def __init__(self):
        self.models = {}
        # TODO: uncomment this expiration time
        # self.expiration_time = 3600 # 1 hour
        self.expiration_time = 6
        self.lock = threading.Lock()
        self.start_expiration_thread()

    def add_model(self, model_id, model):
        with self.lock:
            self.models[model_id] = {'model': model, 'last_used': time.time()}

    def add_model_settings(self, model_id, model_settings):
        with self.lock:
            if model_id in self.models:
                self.models[model_id]['settings'] = model_settings
            else:
                raise KeyError(f"Model ID '{model_id}' does not exist.")

    def get_model(self, model_id):
        with self.lock:
            if model_id in self.models:
                self.models[model_id]['last_used'] = time.time()
                return self.models[model_id]
            else:
                return None

    def remove_model(self, model_id):
        """Remove a model with the given ID."""
        with self.lock:
            if model_id in self.models:
                del self.models[model_id]
            else:
                raise KeyError(f"Model ID '{model_id}' does not exist.")

    def expire_models(self):
        """Remove models that have not been used within the expiration time."""
        current_time = time.time()
        with self.lock:
            to_expire = [
                key for key, value in self.models.items()
                if current_time - value['last_used'] > self.expiration_time
            ]
        for model_id in to_expire:
            try:
                self.remove_model(model_id)
            except Exception as e:
                print(f"Error while expiring model {model_id}: {e}")

    def start_expiration_thread(self):
        def run():
            while True:
                # TODO: uncomment this sleep time
                # time.sleep(600) # 10 minutes
                time.sleep(2)
                print("cleanup models, current models are:")
                print([key for key, value in self.models.items()])
                self.expire_models()
        threading.Thread(target=run, daemon=True).start()
