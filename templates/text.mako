---
outline: [2,3]
---
<%
import pdoc
import re
import textwrap
import inspect
import tomllib

with open('pyproject.toml', 'rb') as _f:
  _project_version = tomllib.load(_f)['project']['version']

def firstline(ds):
  return ds.split('\n\n', 1)[0]

def link(dobj, name=None):
  parts = dobj.qualname.split('.')
  if name is None:
    display = parts[-1]
    if isinstance(dobj, pdoc.Function):
      display += '()'
  else:
    display = name
  if len(parts) >= 2 and parts[0] != 'opengradient':
    return '`{}`'.format(display)
  if len(parts) < 2:
    return '`{}`'.format(display)
  module = parts[1]
  if len(parts) > 2:
    return '[{}](./{})'.format(display, parts[2])
  if isinstance(dobj, pdoc.Module) and dobj.is_package:
    return '[**{}**](./{}/index)'.format(display, module)
  return '[**{}**](./{})'.format(display, module)

def get_annotation(bound_method, sep=':', link=None):
  annot = show_type_annotations and bound_method(link=link) or ''
  if annot:
    annot = ' ' + sep + '\N{NBSP}' + annot
  return annot

def header(text, level):
  return '\n{} {}'.format('#' * level, text)

def breakdown_google(text):
  def get_terms(body):
    parts = re.compile(r'\n+\s+(\w+(?:\s*\([^)]*\))?):\s+', re.MULTILINE).split('\n' + body)
    return list(map(textwrap.dedent, parts[1:]))
  matches = re.compile(r'([A-Z]\w+):$\n', re.MULTILINE).split(inspect.cleandoc(text))
  if not matches:
    return None
  body = textwrap.dedent(matches[0].strip())
  sections = {}
  for i in range(1, len(matches), 2):
    title = matches[i].title()
    section = matches[i+1]
    if title in ('Args', 'Attributes', 'Raises'):
      sections[title] = get_terms(section)
    else:
      sections[title] = textwrap.dedent(section)
  return (body, sections)

def format_for_list(docstring, depth=1):
  spaces = depth * 2 * ' '
  return re.compile(r'\n\n', re.MULTILINE).sub('\n\n' + spaces, docstring)

def resolve_namespace_target(v):
  """If v's type resolves to an opengradient class in a different module, return link target."""
  result = [None]
  var_mod = v.module.name if hasattr(v, 'module') and v.module else ''
  cur_parts = var_mod.split('.')
  def capture(dobj):
    if not isinstance(dobj, pdoc.Module) and hasattr(dobj, 'module') and dobj.module is not None:
      mod_parts = dobj.module.name.split('.')
      if (len(mod_parts) >= 2 and mod_parts[0] == 'opengradient'
          and dobj.module.name != var_mod):
        common = sum(1 for a, b in zip(cur_parts, mod_parts) if a == b)
        remaining = mod_parts[common:]
        if remaining:
          result[0] = '/'.join(remaining)
    return '`{}`'.format(dobj.qualname.split('.')[-1])
  if show_type_annotations:
    v.type_annotation(link=capture)
  return result[0]

def get_return_type_attrs(f):
  """Get Attributes sections from return type class docstrings."""
  import typing
  try:
    hints = typing.get_type_hints(f.obj)
    ret = hints.get('return')
  except Exception:
    return []
  if ret is None:
    return []
  types_to_check = []
  origin = getattr(ret, '__origin__', None)
  if origin is typing.Union:
    types_to_check = list(ret.__args__)
  else:
    types_to_check = [ret]
  results = []
  for t in types_to_check:
    if not isinstance(t, type):
      continue
    fqn = t.__module__ + '.' + t.__qualname__
    dobj = f.module.find_ident(fqn)
    if dobj is None:
      dobj = f.module.find_ident(t.__qualname__)
    if dobj is None or not dobj.docstring:
      continue
    bd = breakdown_google(dobj.docstring)
    if bd and bd[1].get('Attributes'):
      results.append((t.__name__, bd[1]['Attributes']))
  return results

def linkify(text, mod):
  """Convert backtick-wrapped qualified names to markdown links."""
  cur_parts = mod.name.split('.')
  def replace_ref(match):
    name = match.group(1)
    dobj = mod.find_ident(name)
    if dobj is None:
      return match.group(0)
    target_parts = name.split('.')
    # For non-module objects (classes, functions), drop the object name
    # and link to the containing module page instead
    if not isinstance(dobj, pdoc.Module):
      target_parts = target_parts[:-1]
    common = sum(1 for a, b in zip(cur_parts, target_parts) if a == b)
    remaining = target_parts[common:]
    if not remaining:
      return match.group(0)
    display = match.group(2)
    rel_path = '/'.join(remaining)
    target_mod = dobj if isinstance(dobj, pdoc.Module) else dobj.module
    if isinstance(target_mod, pdoc.Module) and target_mod.is_package:
      rel_path += '/index'
    return '[{}](./{})'.format(display, rel_path)
  return re.sub(r'`(opengradient(?:\.\w+)+\.(\w+))`', replace_ref, text)

def breadcrumb(module):
  parts = module.name.split('.')
  if len(parts) <= 1:
    return parts[0]
  current_dir_level = len(parts) - 1 if module.is_package else len(parts) - 2
  crumbs = []
  for i, part in enumerate(parts):
    if i == len(parts) - 1:
      crumbs.append(part)
    else:
      levels_up = current_dir_level - i
      if levels_up == 0:
        path = './index'
      else:
        path = '../' * levels_up + 'index'
      crumbs.append('[{}]({})'.format(part, path))
  return ' / '.join(crumbs)
%>\
<%def name="show_term_list(terms)">\
% for i in range(0, len(terms), 2):
* **`${terms[i]}`**: ${terms[i+1]}
% endfor
</%def>\
<%def name="show_sections(sections, ret_type_attrs=None)">\
% if sections.get('Args'):

**Arguments**

${show_term_list(sections['Args'])}\
% endif
% if sections.get('Attributes'):

**Attributes**

${show_term_list(sections['Attributes'])}\
% endif
% if sections.get('Returns') or ret_type_attrs:

**Returns**

% if sections.get('Returns'):
${sections['Returns']}
% endif
% if ret_type_attrs:
% for type_name, attrs in ret_type_attrs:
**`${type_name}` fields:**

${show_term_list(attrs)}\
% endfor
% endif
% endif
% if sections.get('Raises'):

**Raises**

${show_term_list(sections['Raises'])}\
% endif
% if sections.get('Note'):

**Note**

${sections['Note']}
% endif
% if sections.get('Notes'):

**Notes**

${sections['Notes']}
% endif
</%def>\
<%def name="show_desc(d, short=False, ret_type_attrs=None)">\
<%
inherits = d.inherits
%>\
% if inherits:
_Inherited from:_
% if hasattr(inherits, 'cls'):
`${link(inherits.cls)}`.`${link(inherits, d.name)}`
% else:
`${link(inherits)}`
% endif
% endif
% if short or inherits:
${firstline(d.docstring)}
% elif d.docstring:
<%
bd = breakdown_google(d.docstring)
%>\
% if bd:
${bd[0]}
${show_sections(bd[1], ret_type_attrs=ret_type_attrs)}\
% else:
${d.docstring}
% endif
% endif
</%def>\
<%def name="show_func(f, qual='', level=3)">\
<%
params_list = f.params(annotate=show_type_annotations, link=link)
return_type = get_annotation(f.return_annotation, '\N{non-breaking hyphen}>', link=link)
prefix = qual + ' ' if qual else ''
oneline = '{}{} {}({}){}'.format(prefix, f.funcdef(), f.name, ', '.join(params_list), return_type)
if len(params_list) > 3 or len(oneline) > 88:
  params_block = ',\n    '.join(params_list)
  sig = '{}{} {}(\n    {}\n){}'.format(prefix, f.funcdef(), f.name, params_block, return_type)
else:
  sig = oneline
ret_attrs = get_return_type_attrs(f)
%>

---

${header('`' + f.name + '()`', level)}

```python
${sig}
```
${show_desc(f, ret_type_attrs=ret_attrs)}\
</%def>\
<%def name="show_funcs(fs, qual='', level=3)">\
% for f in fs:
${show_func(f, qual, level)}
% endfor
</%def>\
<%def name="show_vars(vs, qual='')">\
<%
prefix = qual + ' ' if qual else ''
%>\
% for v in vs:
<%
raw_desc = v.docstring.strip() if v.docstring else ''
if raw_desc == 'The type of the None singleton.':
  raw_desc = ''
ns_target = resolve_namespace_target(v)
%>\
% if ns_target:
* [**`${v.name}`**](./${ns_target})${ ': ' + firstline(raw_desc) if raw_desc else ''}
% else:
<%
return_type = get_annotation(v.type_annotation, link=link)
type_str = return_type if return_type else ''
desc = ' - ' + format_for_list(raw_desc, 1) if raw_desc else ''
%>\
* ${prefix}`${v.name}`${type_str}${desc}
% endif
% endfor
</%def>\
<%def name="show_module(module)">\
<%
variables = module.variables(sort=sort_identifiers)
classes = module.classes(sort=sort_identifiers)
functions = module.functions(sort=sort_identifiers)
submodules = module.submodules()
%>
${breadcrumb(module)}
${header('Package ' + module.name, 1)}
% if not module.supermodule:

**Version: ${_project_version}**
% endif

${linkify(module.docstring, module)}
% if submodules:

${header('Submodules', 2)}

% for m in submodules:
* ${link(m)}: ${firstline(m.docstring)}
% endfor
% endif
% if functions:

${header('Functions', 2)}

${show_funcs(functions)}\
% endif
% if variables:

${header('Global variables', 2)}

${show_vars(variables)}\
% endif
% if classes:

${header('Classes', 2)}
% for c in classes:
<%
class_vars = c.class_variables(show_inherited_members, sort=sort_identifiers)
smethods = c.functions(show_inherited_members, sort=sort_identifiers)
inst_vars = c.instance_variables(show_inherited_members, sort=sort_identifiers)
all_methods = c.methods(show_inherited_members, sort=sort_identifiers)
methods = [m for m in all_methods if m.name != '__init__']
subclasses = c.subclasses()
is_enum = any(cls.qualname.split('.')[-1] in ('Enum', 'IntEnum', 'StrEnum', 'Flag', 'IntFlag') for cls in c.mro())
_init_params_list = c.params(annotate=show_type_annotations, link=link)
_init_oneline = 'def __init__({})'.format(', '.join(_init_params_list))
if len(_init_params_list) > 3 or len(_init_oneline) > 88:
  init_sig = 'def __init__(\n    {}\n)'.format(',\n    '.join(_init_params_list))
else:
  init_sig = _init_oneline

_bd = breakdown_google(c.docstring) if c.docstring else None
_class_args = None
if _bd:
  _class_body = _bd[0]
  _class_sections = dict(_bd[1])
  _class_args = _class_sections.pop('Args', None)
else:
  _class_body = c.docstring or ''
  _class_sections = {}

if _class_body.strip().startswith(c.name + '('):
  _class_body = ''
%>
${header('`' + c.name + '`', 3)}
% if _class_body:

${_class_body}
% endif
${show_sections(_class_sections)}\
% if not is_enum and _init_params_list:

${header('Constructor', 4)}

```python
${init_sig}
```
% if _class_args:

**Arguments**

${show_term_list(_class_args)}\
% endif
% endif
% if subclasses:

${header('Subclasses', 4)}

% for sub in subclasses:
* ${link(sub)}
% endfor
% endif
% if smethods:

${header('Static methods', 4)}

${show_funcs(smethods, 'static', 4)}\
% endif
% if methods:

${header('Methods', 4)}

${show_funcs(methods, '', 4)}\
% endif
% if class_vars or inst_vars:

${header('Variables', 4)}

% if class_vars:
${show_vars(class_vars, 'static')}\
% endif
% if inst_vars:
${show_vars(inst_vars)}\
% endif
% endif
% if not show_inherited_members:
<%
members = c.inherited_members()
%>\
% if members:

${header('Inherited members', 4)}

% for cls, mems in members:
* `${link(cls)}`:
% for m in mems:
  * `${link(m, name=m.name)}`
% endfor
% endfor
% endif
% endif
% endfor
% endif
</%def>\
${show_module(module)}
