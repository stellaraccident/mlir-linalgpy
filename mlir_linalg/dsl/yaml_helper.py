"""YAML serialization is routed through here, allowing some things to be
better controlled."""

import yaml


class YAMLObject(yaml.YAMLObject):

  @classmethod
  def to_yaml(cls, dumper, self):
    """Default to a custom dictionary mapping."""
    return dumper.represent_mapping(cls.yaml_tag, self.to_yaml_custom_dict())

  def to_yaml_custom_dict(self):
    raise NotImplementedError()

  def as_linalg_yaml(self):
    return yaml.dump(self)


def dump(data):
  return yaml.dump(data)
