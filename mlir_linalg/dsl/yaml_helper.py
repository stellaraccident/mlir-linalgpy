"""YAML serialization is routed through here, allowing some things to be
better controlled."""

import yaml

__all__ = [
    "yaml_dump",
    "yaml_dump_all",
    "YAMLObject",
]


class YAMLObject(yaml.YAMLObject):

  @classmethod
  def to_yaml(cls, dumper, self):
    """Default to a custom dictionary mapping."""
    return dumper.represent_mapping(cls.yaml_tag, self.to_yaml_custom_dict())

  def to_yaml_custom_dict(self):
    raise NotImplementedError()

  def as_linalg_yaml(self):
    return yaml_dump(self)


def multiline_str_representer(dumper, data):
  if len(data.splitlines()) > 1:
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
  else:
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


yaml.add_representer(str, multiline_str_representer)


def yaml_dump(data, sort_keys=False, **kwargs):
  return yaml.dump(data, sort_keys=sort_keys, **kwargs)


def yaml_dump_all(data, sort_keys=False, **kwargs):
  return yaml.dump(data, sort_keys=sort_keys, **kwargs)
