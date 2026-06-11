# -*- coding: utf-8 -*-

import threading

print_lock = threading.Lock()

def safe_print(msg):
    with print_lock:
        try:
            import readline
            current_input = readline.get_line_buffer()
        except ImportError:
            try:
                import pyreadline3 as readline
                current_input = readline.get_line_buffer()
            except ImportError:
                print(f"\n{msg}")
                print("> ", end="", flush=True)
                return

        print('\r' + ' ' * (len(current_input) + 2), end='')
        print('\r' + msg)
        print(f'\r> {current_input}', end='', flush=True)
