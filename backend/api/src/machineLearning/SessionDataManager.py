import time
import threading

class SessionDataManager:
    def __init__(self):
        self.sessions = {}
        # TODO: uncomment this expiration time
        # self.expiration_time = 3600 # 1 hour
        self.expiration_time = 6
        self.lock = threading.Lock()
        self._start_expiration_thread()

    def add_dataset(self, session_id, dataset):
        self._add_session_data(session_id, 'dataset', dataset)
    
    def add_model(self, session_id, model):
        self._add_session_data(session_id, 'model', model)

    def add_model_settings(self, session_id, model_settings):
        self._add_session_data(session_id, 'model_settings', model_settings)

    def _add_session_data(self, session_id, key, value):
        """This is a private method, do not use directly"""
        if key not in ['dataset', 'model', 'model_settings']:
            raise ValueError(f"Not allowed to add key '{key}' data to the session")
        with self.lock:
            last_used = time.time()
            if session_id not in self.sessions:
                self.sessions[session_id] = {}
            self.sessions[session_id][key] = value
            self.sessions[session_id]['last_used'] = last_used

    def get_session_data(self, session_id):
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]['last_used'] = time.time()
                return self.sessions[session_id]
            else:
                return None

    def remove_session_data(self, session_id):
        """Remove data for a given session ID."""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
            else:
                raise KeyError(f"Data for session ID '{session_id}' does not exist.")

    def _expire_session_data(self):
        """Remove session data that has not been used within the expiration time."""
        current_time = time.time()
        with self.lock:
            to_expire = [
                key for key, value in self.sessions.items()
                if current_time - value['last_used'] > self.expiration_time
            ]
        for session_id in to_expire:
            try:
                self.remove_session_data(session_id)
            except Exception as e:
                print(f"Error while expiring session {session_id}: {e}")

    def _start_expiration_thread(self):
        def run():
            while True:
                # TODO: uncomment this sleep time
                # time.sleep(600) # 10 minutes
                time.sleep(2)
                print("cleanup sessions, current sessions are:")
                print([key for key, value in self.sessions.items()])
                self._expire_session_data()
        threading.Thread(target=run, daemon=True).start()
