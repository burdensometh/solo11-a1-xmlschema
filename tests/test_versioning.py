#!/usr/bin/env python
#
# Copyright (c), 2016-2026, SISSA (International School for Advanced Studies).
# All rights reserved.
# This file is distributed under the terms of the MIT License.
# See the file 'LICENSE' in the root directory of the present
# distribution, or http://opensource.org/licenses/MIT.
#
# @author Davide Brunato <brunato@sissa.it>
#
"""
Tests for content-based schema fingerprints, stale detection and the atomic
reload machinery (``xmlschema.versioning`` and ``XMLSchemaBase.reload``).
"""
import os
import pathlib
import tempfile
import time
import unittest

from xmlschema import XMLSchema10, XMLSchema11, XMLSchemaReloadError, SchemaRegistry
from xmlschema.testing import XMLSchemaTestCase
from xmlschema.versioning import (
    SourceFingerprint, SchemaFingerprint,
    fingerprint_schemas, resource_is_stale,
)

V1_XSD = """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           targetNamespace="http://xmlschema.test/versioning"
           xmlns:tns="http://xmlschema.test/versioning">
  <xs:element name="root">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="name" type="xs:string"/>
      </xs:sequence>
    </xs:complexType>
  </xs:element>
</xs:schema>
"""

V2_XSD = """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           targetNamespace="http://xmlschema.test/versioning"
           xmlns:tns="http://xmlschema.test/versioning">
  <xs:element name="root">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="name" type="xs:string"/>
        <xs:element name="code" type="xs:integer"/>
      </xs:sequence>
    </xs:complexType>
  </xs:element>
</xs:schema>
"""

V1_XML = ('<tns:root xmlns:tns="http://xmlschema.test/versioning">'
          '<name>alpha</name></tns:root>')
V2_XML = ('<tns:root xmlns:tns="http://xmlschema.test/versioning">'
          '<name>alpha</name><code>42</code></tns:root>')

INVALID_XSD = """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:element name="root" type="xs:NOT-A-TYPE"/>
</xs:schema>
"""


class TestFingerprints(unittest.TestCase):

    def test_fingerprint_is_deterministic_and_content_based(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / 'schema.xsd'
            path.write_text(V1_XSD)

            schema = XMLSchema10(str(path))
            fp1 = schema.fingerprint
            fp2 = fingerprint_schemas(schema)

            self.assertIsInstance(fp1, SchemaFingerprint)
            self.assertEqual(fp1.digest, fp2.digest)
            self.assertTrue(all(isinstance(f, SourceFingerprint) for f in fp1.sources))
            # The library's predefined schemas are not part of the fingerprint:
            # only the user's schema remains.
            self.assertEqual(len(fp1.sources), 1)
            self.assertIn('schema.xsd', fp1.sources[0].url)
            self.assertEqual(schema.schema_version, fp1.short)

            # Same content read through another instance -> same digest
            other = XMLSchema10(str(path))
            self.assertEqual(other.fingerprint.digest, fp1.digest)

            # Cached on the instance, recomputed only after a reload
            self.assertIs(schema.fingerprint, fp1)

    def test_fingerprint_changes_with_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / 'schema.xsd'
            path.write_text(V1_XSD)
            schema = XMLSchema10(str(path))
            digest_v1 = schema.fingerprint.digest

            time.sleep(0.02)
            path.write_text(V2_XSD)
            os.utime(path, None)

            # The fingerprint of the *current sources* differs; the cached
            # fingerprint on the stale instance is untouched until reload.
            self.assertNotEqual(fingerprint_schemas(schema).digest, digest_v1)
            self.assertEqual(schema.fingerprint.digest, digest_v1)

    def test_resource_stale_detection_stat_and_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / 'schema.xsd'
            path.write_text(V1_XSD)
            schema = XMLSchema10(str(path))
            recorded = {f.url: f for f in schema.fingerprint.sources}[schema.source.url]

            self.assertFalse(resource_is_stale(schema.source, recorded))
            self.assertFalse(schema.is_stale())

            time.sleep(0.02)
            path.write_text(V2_XSD)
            os.utime(path, None)

            self.assertTrue(resource_is_stale(schema.source, recorded))
            self.assertTrue(resource_is_stale(schema.source, recorded, check_content=True))
            self.assertTrue(schema.is_stale())
            self.assertTrue(schema.is_stale(check_content=True))

    def test_in_memory_resource_fingerprint(self):
        schema = XMLSchema10(V1_XSD)
        fp = fingerprint_schemas(schema)
        self.assertTrue(fp.sources)
        # Re-serializing identical trees produces the same digest
        self.assertEqual(fingerprint_schemas(XMLSchema10(V1_XSD)).digest, fp.digest)


class TestSchemaReload(XMLSchemaTestCase):
    cases_dir = pathlib.Path(__file__).absolute().parent.joinpath('test_cases')

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmpdir.name) / 'schema.xsd'
        self.path.write_text(V1_XSD)
        self.schema = XMLSchema10(str(self.path))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_reload_picks_up_new_rules(self):
        # Force the fingerprint to be computed against the current sources
        # before the upstream file is replaced.
        digest_v1 = self.schema.fingerprint.digest
        self.assertTrue(self.schema.is_valid(V1_XML))
        self.assertFalse(self.schema.is_valid(v2 := V2_XML))

        time.sleep(0.02)
        self.path.write_text(V2_XSD)
        os.utime(self.path, None)

        self.assertTrue(self.schema.is_stale())
        self.assertTrue(self.schema.reload())
        self.assertFalse(self.schema.is_stale())
        self.assertNotEqual(self.schema.fingerprint.digest, digest_v1)

        # New rules: the formerly valid document is now missing 'code'
        self.assertFalse(self.schema.is_valid(V1_XML))
        self.assertTrue(self.schema.is_valid(v2))

        # No changes -> no reload; force always reloads
        self.assertFalse(self.schema.reload())
        self.assertTrue(self.schema.reload(force=True))
        self.assertTrue(self.schema.built)

    def test_reload_is_atomic_and_updates_cached_references(self):
        cached = self.schema  # application keeps a long-lived reference
        digest_v1 = cached.fingerprint.digest

        time.sleep(0.02)
        self.path.write_text(V2_XSD)
        os.utime(self.path, None)
        self.assertTrue(cached.reload())
        self.assertNotEqual(cached.fingerprint.digest, digest_v1)

        # Object identity is preserved while the underlying maps are replaced
        self.assertIs(cached, self.schema)
        self.assertTrue(cached.built)
        self.assertEqual(cached.validation_attempted, 'full')
        self.assertTrue(cached.is_valid(V2_XML))

        # Every component (nested ones included) references the reloaded schema
        root_element = cached.maps.elements[
            '{http://xmlschema.test/versioning}root'
        ]
        self.assertIs(root_element.schema, cached)
        self.assertIs(root_element.type.schema, cached)
        nested_components = [
            c for c in cached.iter_components()
            if c is not cached and c.target_namespace == cached.target_namespace
        ]
        self.assertTrue(nested_components)
        self.assertTrue(all(c.schema is cached for c in nested_components))

        # A brand new schema can still be built (no parent-chain corruption)
        another = XMLSchema10(str(self.path))
        self.assertTrue(another.is_valid(V2_XML))

    def test_failed_reload_keeps_previous_rules(self):
        digest_before = self.schema.fingerprint.digest
        self.assertTrue(self.schema.is_valid(V1_XML))

        time.sleep(0.02)
        self.path.write_text(INVALID_XSD)
        os.utime(self.path, None)

        with self.assertRaises(XMLSchemaReloadError):
            self.schema.reload()

        # The previous, valid rules keep serving
        self.assertTrue(self.schema.built)
        self.assertTrue(self.schema.is_valid(V1_XML))
        self.assertEqual(self.schema.fingerprint.digest, digest_before)

    def test_in_memory_schema_cannot_reload(self):
        schema = XMLSchema10(V1_XSD)
        with self.assertRaises(XMLSchemaReloadError):
            schema.reload()

    def test_reload_from_rejects_meta_and_wrong_type(self):
        with self.assertRaises(XMLSchemaReloadError):
            XMLSchema10.meta_schema.reload_from(XMLSchema10.meta_schema)

        with self.assertRaises(TypeError):
            self.schema.reload_from(XMLSchema11(str(self.path)))

    def test_reload_with_includes(self):
        included = pathlib.Path(self.tmpdir.name) / 'types.xsd'
        included.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           targetNamespace="http://xmlschema.test/versioning"
           xmlns:tns="http://xmlschema.test/versioning">
  <xs:simpleType name="codeType">
    <xs:restriction base="xs:string"/>
  </xs:simpleType>
</xs:schema>
""")
        main = pathlib.Path(self.tmpdir.name) / 'main.xsd'
        main.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           targetNamespace="http://xmlschema.test/versioning"
           xmlns:tns="http://xmlschema.test/versioning">
  <xs:include schemaLocation="{included.as_uri()}"/>
  <xs:element name="root" type="tns:codeType"/>
</xs:schema>
""")
        schema = XMLSchema10(str(main))
        self.assertTrue(schema.is_valid(
            '<tns:root xmlns:tns="http://xmlschema.test/versioning">abc</tns:root>'))

        urls = {f.url for f in schema.fingerprint.sources}
        self.assertIn(included.as_uri(), urls)

        time.sleep(0.02)
        included.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           targetNamespace="http://xmlschema.test/versioning"
           xmlns:tns="http://xmlschema.test/versioning">
  <xs:simpleType name="codeType">
    <xs:restriction base="xs:integer"/>
  </xs:simpleType>
</xs:schema>
""")
        os.utime(included, None)

        self.assertTrue(schema.is_stale())
        self.assertTrue(schema.reload())
        self.assertFalse(schema.is_valid(
            '<tns:root xmlns:tns="http://xmlschema.test/versioning">abc</tns:root>'))
        self.assertTrue(schema.is_valid(
            '<tns:root xmlns:tns="http://xmlschema.test/versioning">7</tns:root>'))


class TestSchemaRegistry(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmpdir.name) / 'schema.xsd'
        self.path.write_text(V1_XSD)
        self.registry = SchemaRegistry()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_get_is_canonical_and_versioned(self):
        first = self.registry.get(str(self.path))
        second = self.registry.get(self.path.as_uri())
        self.assertIs(first, second)
        key = self.path.as_uri()
        self.assertIn(key, self.registry)
        self.assertEqual(len(self.registry), 1)

        versions = self.registry.versions()
        self.assertEqual(set(versions), {key})
        self.assertEqual(versions[key], first.schema_version)
        self.assertEqual(
            self.registry.fingerprint(str(self.path)).digest,
            first.fingerprint.digest,
        )

    def test_reload_swaps_rules_in_place(self):
        schema = self.registry.get(str(self.path))
        digest_v1 = schema.fingerprint.digest
        self.assertTrue(schema.is_valid(V1_XML))
        self.assertFalse(schema.is_valid(V2_XML))

        time.sleep(0.02)
        self.path.write_text(V2_XSD)
        os.utime(self.path, None)

        self.assertEqual(self.registry.reload(), [self.path.as_uri()])
        # Same object, new rules
        self.assertIs(self.registry.get(str(self.path)), schema)
        self.assertFalse(schema.is_valid(V1_XML))
        self.assertTrue(schema.is_valid(V2_XML))
        self.assertFalse(self.registry.is_stale(str(self.path)))
        self.assertNotEqual(schema.fingerprint.digest, digest_v1)

    def test_reload_failure_keeps_old_instance(self):
        schema = self.registry.get(str(self.path))
        digest_v1 = schema.fingerprint.digest

        time.sleep(0.02)
        self.path.write_text(INVALID_XSD)
        os.utime(self.path, None)

        with self.assertRaises(XMLSchemaReloadError):
            self.registry.reload()
        self.assertTrue(self.registry.get(str(self.path)).is_valid(V1_XML))
        self.assertEqual(self.registry.get(str(self.path)).fingerprint.digest, digest_v1)

    def test_unknown_location_and_clear(self):
        with self.assertRaises(KeyError):
            self.registry.reload('/does/not/exist.xsd')
        with self.assertRaises(KeyError):
            self.registry.fingerprint('/does/not/exist.xsd')

        self.registry.get(str(self.path))
        self.registry.clear()
        self.assertEqual(len(self.registry), 0)

    def test_register_and_get_in_memory_source(self):
        schema = XMLSchema10(V1_XSD)
        key = self.registry.register(schema)
        self.assertIn(key, self.registry)
        self.assertEqual(self.registry.versions()[key], schema.schema_version)

        with self.assertRaises(XMLSchemaReloadError):
            self.registry.get('<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"/>')


if __name__ == '__main__':
    unittest.main()
