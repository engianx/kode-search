from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.formatted_text import HTML

from dataclasses import dataclass

@dataclass
class Entity:
    file: str
    line: int
    content: str
    summary: str

class Viewer:
    def __init__(self, raw_entities):
        self._idx = 0
        self._buffer = Buffer()
        self._entities = self._convert(raw_entities)

        self._content_panel = FormattedTextControl(
            text=self._cur_content())
        self._left_panel = FormattedTextControl(
            text=self._entity_selection())
        
        self._root_view = VSplit([
            Window(content=self._left_panel, width=40),
            Window(content=self._content_panel, width=80)
        ])

        self._layout = Layout(self._root_view)
        self._keybindings = KeyBindings()

        @self._keybindings.add('q')
        def _(event):
            event.app.exit()

        @self._keybindings.add('down')
        def _(event):
            self._go_down()

        @self._keybindings.add('up')
        def _(event):
            self._go_up()

        self._app = Application(
            layout=self._layout, full_screen=False, key_bindings=self._keybindings)

    def _convert(self, raw_entities):
        return [Entity(e.get('file'), e.get('start_line'), e.get('content'), e.get('summary')) for e in raw_entities]

    def _entity_selection(self):
        lines = []
        for i, entity in enumerate(self._entities):
            if i == self._idx:
                lines.append(to_formatted_text(HTML(
                    f'👉 {entity.file.split("/")[-1:][0]}:{entity.line}'), style='#7474FF'))
            else:
                lines.append([('', 
                    f'   {entity.file.split("/")[-1:][0]}:{entity.line}')])
            lines.append([('', '\n')])

        return [item for sublist in lines for item in sublist]

    def _cur_content(self):
        entity = self._entities[self._idx]
        if entity.summary is None:
            return entity.content
        return entity.summary + '\n\n' + entity.content
    
    def _go_down(self):
        if self._idx < len(self._entities) - 1:
            self._idx += 1
            self._content_panel.text = self._cur_content()
            self._left_panel.text = self._entity_selection()
    
    def _go_up(self):
        if self._idx > 0:
            self._idx -= 1
            self._content_panel.text = self._cur_content()
            self._left_panel.text = self._entity_selection()

    def run(self):
        self._app.run()