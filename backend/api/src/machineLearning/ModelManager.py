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

    def add_model_settings(self, model_id, modelSettings):
        with self.lock:
            if model_id in self.models:
                self.models[model_id]['settings'] = modelSettings
            else:
                raise KeyError(f"Model ID '{model_id}' does not exist.")

    def get_model(self, model_id):
        with self.lock:
            if model_id in self.models:
                self.models[model_id]['last_used'] = time.time()
                return self.models[model_id]
            else:
                return None

    def expire_models(self):
        with self.lock:
            current_time = time.time()
            to_expire = [
                key for key, value in self.models.items()
                if current_time - value['last_used'] > self.expiration_time
            ]
            for key in to_expire:
                del self.models[key]

    def start_expiration_thread(self):
        def run():
            while True:
                # TODO: uncomment this sleep time
                # time.sleep(600) # 10 minutes
                time.sleep(2)
                print("cleanup models")
                self.expire_models()
        threading.Thread(target=run, daemon=True).start()
