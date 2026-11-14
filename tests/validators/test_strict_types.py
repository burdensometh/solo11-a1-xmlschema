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
"""Tests for the switchable strict type checking of the encoding process."""
import unittest
from collections import Counter
from textwrap import dedent

from elementpath.datatypes import Date10

from xmlschema import XMLSchema10, XMLSchema11, XMLSchemaEncodeError
from xmlschema.validators.validation import EncodeContext


SCHEMA = dedent("""\
    <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
      <xs:element name="root">
        <xs:complexType>
          <xs:sequence>
            <xs:element name="count" type="xs:int"/>
            <xs:element name="flag" type="xs:boolean"/>
            <xs:element name="ratio" type="xs:decimal"/>
            <xs:element name="name" type="xs:token"/>
            <xs:element name="when" type="xs:date"/>
          </xs:sequence>
          <xs:attribute name="code" type="xs:integer"/>
          <xs:attribute name="active" type="xs:boolean"/>
          <xs:attribute name="label" type="xs:string"/>
        </xs:complexType>
      </xs:element>
    </xs:schema>""")

DATA = {
    'count': '',           # empty string for a non string-based type
    'flag': 1,             # Python int used as a boolean
    'ratio': '3.14',       # lexical string coerced to decimal
    'name': 42,            # Python int coerced to token
    'when': '2026-11-14',  # lexical string coerced to date
    '@code': '7',          # lexical string coerced to integer
    '@active': True,       # native boolean, no conversion expected
    '@label': '',          # empty string is legal for xs:string
}


class TestStrictTypesEncoding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.schema = XMLSchema10(SCHEMA)

    def test_default_mode_is_lenient_and_counts_conversions(self):
        counter = Counter()
        _, errors = self.schema.encode(
            DATA, validation='lax', implicit_conversions=counter
        )
        # Existing behavior: type mismatches are signaled but data is converted
        self.assertTrue(errors)
        self.assertEqual(counter['count (empty string to int)'], 1)
        self.assertEqual(counter['flag (integer to boolean)'], 1)
        self.assertEqual(counter['ratio (string to decimal)'], 1)
        self.assertEqual(counter['name (int to token)'], 1)
        self.assertEqual(counter['when (string to date)'], 1)
        self.assertEqual(counter['@code (string to integer)'], 1)

        keys = ''.join(counter)
        self.assertNotIn('@label', keys)   # xs:string accepts the empty string
        self.assertNotIn('@active', keys)  # native bool is not a conversion

    def test_strict_mode_collects_errors_with_field_names(self):
        _, errors = self.schema.encode(DATA, validation='lax', strict_types=True)
        reasons = '\n'.join(err.reason for err in errors)

        # Empty string and Python type mismatches are rejected ...
        self.assertIn('field count', reasons)
        self.assertIn('empty string', reasons)
        self.assertIn('field flag', reasons)
        self.assertIn('field name', reasons)

        # ... while lexical strings and native values remain legal
        self.assertNotIn('field ratio', reasons)
        self.assertNotIn('field when', reasons)
        self.assertNotIn('field @code', reasons)
        self.assertNotIn('@label', reasons)

    def test_strict_mode_raises_with_strict_validation(self):
        with self.assertRaises(XMLSchemaEncodeError) as ctx:
            self.schema.encode(DATA, validation='strict', strict_types=True)
        self.assertIn('field', ctx.exception.reason)

    def test_strict_mode_accepts_native_values(self):
        clean = {
            'count': 5,
            'flag': True,
            'ratio': 3.14,
            'name': 'abc',
            'when': Date10.fromstring('2026-11-14'),
            '@code': 7,
            '@active': False,
            '@label': '',
        }
        counter = Counter()
        _, errors = self.schema.encode(
            clean, validation='lax', strict_types=True,
            implicit_conversions=counter
        )
        self.assertEqual(errors, [])
        self.assertEqual(dict(counter), {})

    def test_boolean_lexical_representations_remain_legal(self):
        data = {
            'count': 1, 'flag': 'true', 'ratio': 1.5, 'name': 'x',
            'when': Date10.fromstring('2020-01-01'), '@active': '1'
        }
        _, errors = self.schema.encode(data, validation='lax', strict_types=True)
        self.assertEqual(errors, [])

        # but a Python integer must still be rejected
        _, errors = self.schema.encode(
            {**data, 'flag': 1}, validation='lax', strict_types=True
        )
        self.assertTrue(any('field flag' in e.reason for e in errors))

    def test_empty_string_for_string_type_is_legal(self):
        _, errors = self.schema.encode(
            {'count': 0, 'flag': False, 'ratio': 1.0, 'name': 'n',
             'when': Date10.fromstring('2020-01-01'), '@label': ''},
            validation='lax', strict_types=True
        )
        self.assertFalse(any('@label' in e.reason for e in errors))

    def test_list_type_empty_item(self):
        schema = XMLSchema10(dedent("""\
            <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
              <xs:element name="nums">
                <xs:simpleType><xs:list itemType="xs:int"/></xs:simpleType>
              </xs:element>
            </xs:schema>"""))

        _, errors = schema.encode([1, '', 3], validation='lax', strict_types=True)
        self.assertTrue(any('empty string' in e.reason for e in errors))

        counter = Counter()
        schema.encode([1, '', 3], validation='lax', implicit_conversions=counter)
        self.assertEqual(counter['nums (empty string to int)'], 1)

    def test_boolean_value_for_non_boolean_type(self):
        data = {'count': True, 'flag': False, 'ratio': 1.0, 'name': 'n',
                'when': Date10.fromstring('2020-01-01')}

        _, errors = self.schema.encode(data, validation='lax', strict_types=True)
        self.assertTrue(any('field count' in e.reason for e in errors))

        counter = Counter()
        self.schema.encode(data, validation='lax', implicit_conversions=counter)
        self.assertEqual(counter['count (boolean to int)'], 1)

    def test_union_type_strict_mode(self):
        schema = XMLSchema10(dedent("""\
            <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
              <xs:element name="v">
                <xs:simpleType>
                  <xs:union memberTypes="xs:integer xs:decimal"/>
                </xs:simpleType>
              </xs:element>
            </xs:schema>"""))

        # empty string matches neither member in strict mode
        _, errors = schema.encode('', validation='lax', strict_types=True)
        self.assertTrue(errors)

        # a string member makes the empty string a legal lexical value
        schema_with_string = XMLSchema10(dedent("""\
            <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
              <xs:element name="v">
                <xs:simpleType>
                  <xs:union memberTypes="xs:integer xs:string"/>
                </xs:simpleType>
              </xs:element>
            </xs:schema>"""))
        _, errors = schema_with_string.encode(
            '', validation='lax', strict_types=True
        )
        self.assertEqual(errors, [])

    def test_xsd11_strict_mode(self):
        schema = XMLSchema11(dedent("""\
            <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
              <xs:element name="root">
                <xs:complexType>
                  <xs:sequence>
                    <xs:element name="n" type="xs:int"/>
                    <xs:element name="b" type="xs:boolean"/>
                  </xs:sequence>
                  <xs:assert test="xs:int(n) >= 0"/>
                </xs:complexType>
              </xs:element>
            </xs:schema>"""))

        counter = Counter()
        _, errors = schema.encode(
            {'n': '5', 'b': 0}, validation='lax', implicit_conversions=counter
        )
        self.assertEqual(counter['n (string to int)'], 1)
        self.assertEqual(counter['b (integer to boolean)'], 1)

        _, errors = schema.encode(
            {'n': '5', 'b': 0}, validation='lax', strict_types=True
        )
        self.assertTrue(any('field b' in e.reason for e in errors))

        _, errors = schema.encode(
            {'n': 5, 'b': True}, validation='lax', strict_types=True
        )
        self.assertEqual(errors, [])


class TestEncodeContextCounters(unittest.TestCase):

    def test_context_keeps_an_injected_counter(self):
        counter = Counter()
        context = EncodeContext(source={}, implicit_conversions=counter)
        self.assertIs(context.implicit_conversions, counter)

    def test_field_name_resolution(self):
        from xml.etree import ElementTree as ET

        context = EncodeContext(source={})
        self.assertIsNone(context.get_field_name())

        context.attribute = '{http://example.com/ns}code'
        self.assertTrue(context.get_field_name().startswith('@'))

        context.attribute = None
        context.elem = ET.Element('{http://example.com/ns}root')
        self.assertEqual(context.get_field_name(), '{http://example.com/ns}root')

    def test_report_formatting(self):
        context = EncodeContext(source={})
        self.assertEqual(context.format_implicit_conversions(),
                         'no implicit type conversions')

        context.count_implicit_conversion('string to int')
        context.count_implicit_conversion('string to int')
        report = context.format_implicit_conversions()
        self.assertIn('implicit type conversions per field', report)
        self.assertIn('(string to int): 2', report)


if __name__ == '__main__':
    unittest.main()
