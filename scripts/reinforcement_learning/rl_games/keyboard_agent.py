from pynput import keyboard
import time

class KeyboardTeleop:
    def __init__(self):
        self.key_states = {
            "w": False, "s": False,
            "a": False, "d": False,
            "up": False, "down": False,
            "q": False, "e": False,
        }
        self.stop = False
        self.reset = False

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
            suppress=True,
        )
        self.listener.start()

    def on_press(self, key):
        if key == keyboard.Key.esc:
            self.stop = True
            self.listener.stop()
            return

        try:
            kc = key.char.lower()
        except AttributeError:
            if key == keyboard.Key.up:
                self.key_states["up"] = True
            elif key == keyboard.Key.down:
                self.key_states["down"] = True
            return

        if kc in self.key_states:
            self.key_states[kc] = True
        
        if kc == 'r':
            self.reset = True

    def on_release(self, key):
        try:
            kc = key.char.lower()
        except AttributeError:
            if key == keyboard.Key.up:
                self.key_states["up"] = False
            elif key == keyboard.Key.down:
                self.key_states["down"] = False
            return

        if kc in self.key_states:
            self.key_states[kc] = False
            print(f"[INFO] Key '{kc}' released.")

        if kc == 'r':
            self.reset = False

    def get_command(self):
        vx = float(self.key_states["w"]) - float(self.key_states["s"])
        vy = float(self.key_states["a"]) - float(self.key_states["d"])
        vz = float(self.key_states["up"]) - float(self.key_states["down"])
        vg = float(self.key_states["q"])
        return vx, vy, vz, vg, self.stop, self.reset

    def run(self):
        print("Keyboard teleop started. Hold keys to move, ESC to quit.")
        while not self.stop:
            vx, vy, vz, vg, _ = self.get_command()
            print(f"\rvx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, vg={vg:.2f}", end="")
            time.sleep(0.05)
        print("\nTeleop stopped.")

if __name__ == "__main__":
    KeyboardTeleop().run()
