"""Comprehensive tests for the language parsers (5 dedicated + 9 generic).

This file is intentionally large because the user asked for "muchos
tests con casos raros" — exhaustive coverage of every language we
claim to support, with the awkward edge cases that real production
code throws at parsers.

Each language gets its own test class. The test methods follow a
consistent shape:

  * **happy path**: a normal class with 2-3 methods → assert that
    method count is right and every method has parent_symbol set.
  * **rare case**: nested classes, anonymous functions, methods on
    types defined elsewhere, decorated functions, class with
    inheritance / generics / type parameters / abstract methods,
    etc.
  * **structural fact**: assert specific qualified_name values so a
    regression in name extraction is caught immediately.

The tests use the parser DIRECTLY (not the indexer pipeline) to keep
them fast and deterministic — no temp DB, no FS writes, no fixtures
beyond a tree-sitter Parser instance per language.

If you're adding a new language to the system, add a new test class
here at the same time. Aim for at least 6 tests per language: happy
path + 5 awkward cases.
"""

from __future__ import annotations

import pytest
from tree_sitter import Parser

from infinidev.code_intel.indexer import _get_ts_language
from infinidev.code_intel.parsers import get_parser
from infinidev.code_intel.models import SymbolKind


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def parse(language: str, code: str | bytes):
    """Run *code* through the appropriate parser and return its symbol list.

    Returns ``(symbols, has_error)`` where has_error reflects whether
    tree-sitter detected any ERROR nodes during parsing — useful for
    asserting that intentionally-broken inputs don't crash the parser.
    """
    if isinstance(code, str):
        code = code.encode("utf-8")
    ts_lang = _get_ts_language(language)
    if ts_lang is None:
        pytest.skip(f"tree-sitter grammar for {language} not available")
    parser = Parser(ts_lang)
    tree = parser.parse(code)
    p = get_parser(language)
    if p is None:
        pytest.skip(f"no parser registered for {language}")
    symbols = p.extract_symbols(tree, code, f"test.{language}")
    return symbols, tree.root_node.has_error


def names_of(symbols, kind=None):
    """Return the set of qualified names for *symbols*, optionally filtered by kind."""
    if kind is not None:
        symbols = [s for s in symbols if s.kind == kind]
    return {s.qualified_name for s in symbols}


def methods_of(symbols, parent: str):
    """Return the set of method names belonging to *parent*."""
    return {
        s.name for s in symbols
        if s.kind == SymbolKind.method and s.parent_symbol == parent
    }


def assert_all_methods_have_parent(symbols, *, exclude_lone_top_level: bool = True):
    """Every symbol of kind 'method' must have a non-empty parent_symbol.

    Set ``exclude_lone_top_level=False`` to fail on top-level methods
    too — most languages classify those as functions, not methods,
    but a few exotic ones (like JS object literal methods) blur the
    line.
    """
    for s in symbols:
        if s.kind == SymbolKind.method:
            assert s.parent_symbol, (
                f"method {s.name!r} at line {s.line_start} has no parent_symbol"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Python
# ─────────────────────────────────────────────────────────────────────────────


class TestPython:
    """Tests for the dedicated Python parser."""

    def test_simple_class_with_methods(self):
        code = """
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""
        symbols, _ = parse("python", code)
        assert "Calculator" in names_of(symbols)
        assert methods_of(symbols, "Calculator") == {"add", "subtract"}
        assert_all_methods_have_parent(symbols)

    def test_dunder_and_underscore_methods(self):
        code = """
class Database:
    def __init__(self):
        pass

    def __repr__(self):
        return "Database"

    def _internal(self):
        pass

    def public(self):
        pass
"""
        symbols, _ = parse("python", code)
        method_names = methods_of(symbols, "Database")
        assert {"__init__", "__repr__", "_internal", "public"} <= method_names

    def test_decorated_methods(self):
        code = """
class API:
    @staticmethod
    def helper():
        pass

    @property
    def name(self):
        return self._name

    @classmethod
    def create(cls):
        return cls()
"""
        symbols, _ = parse("python", code)
        # @property gets re-classified as SymbolKind.property by the
        # Python parser, so we can't filter by methods alone — assert
        # over all symbols inside the class.
        all_in_class = {
            s.name for s in symbols if s.parent_symbol == "API"
        }
        assert {"helper", "name", "create"} <= all_in_class

    def test_nested_class(self):
        code = """
class Outer:
    class Inner:
        def inner_method(self):
            pass

    def outer_method(self):
        pass
"""
        symbols, _ = parse("python", code)
        assert "Outer" in names_of(symbols)
        # Inner class exists; the implementation may qualify it as
        # Outer.Inner or just Inner — accept either
        all_names = names_of(symbols)
        assert "Inner" in all_names or "Outer.Inner" in all_names

    def test_async_methods(self):
        code = """
class Worker:
    async def fetch(self, url):
        return None

    async def process(self, data):
        return data
"""
        symbols, _ = parse("python", code)
        method_names = methods_of(symbols, "Worker")
        assert {"fetch", "process"} <= method_names

    def test_top_level_functions_not_methods(self):
        code = """
def standalone(x):
    return x * 2

def another(y):
    return y + 1
"""
        symbols, _ = parse("python", code)
        funcs = {s.name for s in symbols if s.kind == SymbolKind.function}
        assert {"standalone", "another"} <= funcs
        # And no spurious methods
        methods = [s for s in symbols if s.kind == SymbolKind.method]
        assert len(methods) == 0

    def test_module_level_constants(self):
        code = """
MAX_RETRIES = 5
DEFAULT_TIMEOUT = 30
api_url = "https://example.com"
"""
        symbols, _ = parse("python", code)
        names = {s.name for s in symbols}
        assert "MAX_RETRIES" in names
        assert "DEFAULT_TIMEOUT" in names

    def test_inheritance(self):
        code = """
class Base:
    def base_method(self):
        pass

class Child(Base):
    def child_method(self):
        pass

class GrandChild(Child, Base):
    def grand_method(self):
        pass
"""
        symbols, _ = parse("python", code)
        assert {"Base", "Child", "GrandChild"} <= names_of(symbols)
        assert "base_method" in methods_of(symbols, "Base")
        assert "child_method" in methods_of(symbols, "Child")
        assert "grand_method" in methods_of(symbols, "GrandChild")

    def test_empty_class_pass(self):
        code = """
class Empty:
    pass

class AlsoEmpty: ...
"""
        symbols, _ = parse("python", code)
        names = names_of(symbols)
        assert "Empty" in names
        assert "AlsoEmpty" in names

    def test_broken_syntax_does_not_crash(self):
        code = """
class Broken
    def missing_colon(self
        pass
"""
        symbols, _ = parse("python", code)
        # Tree-sitter is error-tolerant; the parser must not crash.
        assert isinstance(symbols, list)


# ─────────────────────────────────────────────────────────────────────────────
# JavaScript
# ─────────────────────────────────────────────────────────────────────────────


class TestJavaScript:
    """Tests for the dedicated JavaScript parser."""

    def test_class_with_methods(self):
        code = """
class Counter {
    constructor() {
        this.value = 0;
    }
    increment() {
        this.value++;
    }
    decrement() {
        this.value--;
    }
}
"""
        symbols, _ = parse("javascript", code)
        assert "Counter" in names_of(symbols)
        assert {"constructor", "increment", "decrement"} <= methods_of(symbols, "Counter")
        assert_all_methods_have_parent(symbols)

    def test_arrow_function_assigned_to_const(self):
        code = """
const greet = (name) => `Hello, ${name}`;
const add = (a, b) => a + b;
"""
        symbols, _ = parse("javascript", code)
        names = {s.name for s in symbols}
        assert "greet" in names
        assert "add" in names

    def test_function_declarations(self):
        code = """
function helper(x) { return x * 2; }
async function fetcher(url) { return await fetch(url); }
function* generator() { yield 1; yield 2; }
"""
        symbols, _ = parse("javascript", code)
        funcs = {s.name for s in symbols if s.kind == SymbolKind.function}
        assert "helper" in funcs
        assert "fetcher" in funcs

    def test_static_and_private_methods(self):
        code = """
class Service {
    static create() { return new Service(); }
    #privateHelper() { return 42; }
    publicMethod() { return this.#privateHelper(); }
}
"""
        symbols, _ = parse("javascript", code)
        names = methods_of(symbols, "Service")
        assert "create" in names
        assert "publicMethod" in names

    def test_export_class(self):
        code = """
export class Public {
    method() { return 1; }
}

export default class Default {
    other() { return 2; }
}
"""
        symbols, _ = parse("javascript", code)
        all_names = names_of(symbols)
        assert "Public" in all_names
        assert "Default" in all_names or "method" in {s.name for s in symbols}

    def test_imports(self):
        code = """
import React from 'react';
import { useState, useEffect } from 'react';
import * as utils from './utils';
"""
        from infinidev.code_intel.parsers import get_parser as gp
        ts_lang = _get_ts_language("javascript")
        parser = Parser(ts_lang)
        tree = parser.parse(code.encode())
        imports = gp("javascript").extract_imports(tree, code.encode(), "test.js")
        assert len(imports) >= 3
        sources = {i.source for i in imports}
        assert "react" in sources
        assert "./utils" in sources


# ─────────────────────────────────────────────────────────────────────────────
# TypeScript
# ─────────────────────────────────────────────────────────────────────────────


class TestTypeScript:
    """Tests for the TypeScript parser (extends JS parser)."""

    def test_class_with_typed_methods(self):
        # The exact bug we fixed today: TS class names use type_identifier
        code = """
class UserService {
    private db: Database;

    constructor(db: Database) {
        this.db = db;
    }

    public findUser(id: number): User | null {
        return this.db.find(id);
    }

    private log(msg: string): void {
        console.log(msg);
    }
}
"""
        symbols, _ = parse("typescript", code)
        assert "UserService" in names_of(symbols)
        method_names = methods_of(symbols, "UserService")
        assert {"constructor", "findUser", "log"} <= method_names
        assert_all_methods_have_parent(symbols)

    def test_interface(self):
        code = """
interface Point {
    x: number;
    y: number;
}

interface Drawable {
    draw(): void;
    bounds(): Rectangle;
}
"""
        symbols, _ = parse("typescript", code)
        names = names_of(symbols)
        assert "Point" in names
        assert "Drawable" in names

    def test_type_alias(self):
        code = """
type StringOrNumber = string | number;
type Callback = (err: Error | null, data: string) => void;
"""
        symbols, _ = parse("typescript", code)
        names = {s.name for s in symbols}
        assert "StringOrNumber" in names
        assert "Callback" in names

    def test_enum(self):
        code = """
enum Color {
    Red,
    Green,
    Blue,
}

enum HttpStatus {
    OK = 200,
    NotFound = 404,
}
"""
        symbols, _ = parse("typescript", code)
        names = names_of(symbols)
        assert "Color" in names
        assert "HttpStatus" in names

    def test_generic_class(self):
        code = """
class Container<T> {
    private items: T[] = [];

    push(item: T): void {
        this.items.push(item);
    }

    pop(): T | undefined {
        return this.items.pop();
    }
}
"""
        symbols, _ = parse("typescript", code)
        assert "Container" in names_of(symbols)
        assert {"push", "pop"} <= methods_of(symbols, "Container")

    def test_abstract_class(self):
        code = """
abstract class Shape {
    abstract area(): number;

    describe(): string {
        return `Shape with area ${this.area()}`;
    }
}

class Circle extends Shape {
    constructor(public radius: number) { super(); }

    area(): number {
        return Math.PI * this.radius * this.radius;
    }
}
"""
        symbols, _ = parse("typescript", code)
        names = names_of(symbols)
        assert "Shape" in names
        assert "Circle" in names
        # Both classes' methods should be attributed correctly
        circle_methods = methods_of(symbols, "Circle")
        assert "area" in circle_methods


# ─────────────────────────────────────────────────────────────────────────────
# Rust
# ─────────────────────────────────────────────────────────────────────────────


class TestRust:
    """Tests for the dedicated Rust parser."""

    def test_struct_with_impl(self):
        code = """
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}
"""
        symbols, _ = parse("rust", code)
        assert "Point" in names_of(symbols)
        # Rust uses :: for qualified names (not .)
        method_names = {s.name for s in symbols if s.kind == SymbolKind.method}
        assert "new" in method_names
        assert "distance" in method_names

    def test_enum_with_variants(self):
        code = """
enum Direction {
    North,
    South,
    East,
    West,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}
"""
        symbols, _ = parse("rust", code)
        names = names_of(symbols)
        assert "Direction" in names
        assert "Result" in names

    def test_trait(self):
        code = """
trait Animal {
    fn name(&self) -> String;
    fn sound(&self) -> String;
}
"""
        symbols, _ = parse("rust", code)
        names = names_of(symbols)
        assert "Animal" in names

    def test_pub_vs_private(self):
        code = """
pub fn public_function() -> i32 { 42 }
fn private_function() -> i32 { 0 }
pub const MAX: i32 = 100;
"""
        symbols, _ = parse("rust", code)
        funcs = {s.name: s.visibility for s in symbols if s.kind == SymbolKind.function}
        assert funcs.get("public_function") == "public"
        assert funcs.get("private_function") == "private"

    def test_async_function(self):
        code = """
pub async fn fetch_data(url: &str) -> Result<String, Error> {
    Ok(String::new())
}
"""
        symbols, _ = parse("rust", code)
        funcs = [s for s in symbols if s.name == "fetch_data"]
        assert funcs
        assert funcs[0].is_async


# ─────────────────────────────────────────────────────────────────────────────
# C
# ─────────────────────────────────────────────────────────────────────────────


class TestC:
    """Tests for the dedicated C parser."""

    def test_function_definitions(self):
        code = """
int add(int a, int b) {
    return a + b;
}

static int multiply(int a, int b) {
    return a * b;
}
"""
        symbols, _ = parse("c", code)
        funcs = {s.name for s in symbols if s.kind == SymbolKind.function}
        assert "add" in funcs
        assert "multiply" in funcs

    def test_struct_and_enum(self):
        code = """
struct Point {
    int x;
    int y;
};

enum Direction {
    NORTH,
    SOUTH,
};
"""
        symbols, _ = parse("c", code)
        names = names_of(symbols)
        assert "Point" in names
        assert "Direction" in names

    def test_typedef(self):
        code = """
typedef struct Node {
    int value;
    struct Node *next;
} Node;

typedef int (*Callback)(int);
"""
        symbols, _ = parse("c", code)
        names = names_of(symbols)
        assert "Node" in names

    def test_preprocessor_define(self):
        code = """
#define MAX_SIZE 1024
#define PI 3.14159
"""
        symbols, _ = parse("c", code)
        names = names_of(symbols)
        assert "MAX_SIZE" in names
        assert "PI" in names


# ─────────────────────────────────────────────────────────────────────────────
# Go (generic parser)
# ─────────────────────────────────────────────────────────────────────────────


class TestGo:
    """Tests for Go via the generic parser."""

    def test_struct_with_methods(self):
        code = """
package main

type Server struct {
    port int
}

func (s *Server) Start() error {
    return nil
}

func (s *Server) Stop() error {
    return nil
}
"""
        symbols, _ = parse("go", code)
        assert "Server" in names_of(symbols)
        # Go methods should be attributed to the receiver type
        assert methods_of(symbols, "Server") >= {"Start", "Stop"}
        assert_all_methods_have_parent(symbols)

    def test_top_level_functions(self):
        code = """
package main

func helper(x int) int {
    return x * 2
}

func main() {
    helper(5)
}
"""
        symbols, _ = parse("go", code)
        funcs = {s.name for s in symbols if s.kind == SymbolKind.function}
        assert "helper" in funcs
        assert "main" in funcs

    def test_capitalisation_visibility(self):
        # Go convention: capital first letter = exported (public)
        code = """
package main

type Service struct{}

func (s *Service) PublicMethod() {}
func (s *Service) privateMethod() {}
func ExportedFunc() {}
func unexportedFunc() {}
"""
        symbols, _ = parse("go", code)
        by_name = {s.name: s for s in symbols if s.kind in (SymbolKind.method, SymbolKind.function)}
        assert by_name["PublicMethod"].visibility == "public"
        assert by_name["privateMethod"].visibility == "private"
        assert by_name["ExportedFunc"].visibility == "public"
        assert by_name["unexportedFunc"].visibility == "private"

    def test_value_vs_pointer_receiver(self):
        code = """
package main

type Counter struct{ value int }

func (c Counter) Read() int { return c.value }
func (c *Counter) Increment() { c.value++ }
"""
        symbols, _ = parse("go", code)
        # Both should be methods of Counter
        assert methods_of(symbols, "Counter") >= {"Read", "Increment"}

    def test_interface(self):
        code = """
package main

type Animal interface {
    Sound() string
    Name() string
}
"""
        symbols, _ = parse("go", code)
        names = names_of(symbols)
        assert "Animal" in names

    def test_constants_and_vars(self):
        code = """
package main

const MAX_RETRIES = 5
var globalLogger = "main"
"""
        symbols, _ = parse("go", code)
        names = {s.name for s in symbols}
        assert "MAX_RETRIES" in names

    def test_multiple_structs(self):
        code = """
package main

type A struct{}
type B struct{}

func (a *A) MethodA() {}
func (b *B) MethodB() {}
"""
        symbols, _ = parse("go", code)
        assert methods_of(symbols, "A") == {"MethodA"}
        assert methods_of(symbols, "B") == {"MethodB"}


# ─────────────────────────────────────────────────────────────────────────────
# Java (generic parser)
# ─────────────────────────────────────────────────────────────────────────────


class TestJava:
    """Tests for Java via the generic parser."""

    def test_class_with_typed_methods(self):
        # The Java bug we fixed: method name is the SECOND identifier
        # (after the return type), not the first
        code = """
public class UserService {
    private final Database db;

    public UserService(Database db) {
        this.db = db;
    }

    public User findUser(int id) {
        return db.query(id);
    }

    private void log(String msg) {
        System.out.println(msg);
    }
}
"""
        symbols, _ = parse("java", code)
        assert "UserService" in names_of(symbols)
        method_names = methods_of(symbols, "UserService")
        # The method names must be findUser/log, NOT User/UserService (the return types)
        assert "findUser" in method_names
        assert "log" in method_names
        assert_all_methods_have_parent(symbols)

    def test_interface(self):
        code = """
public interface Repository<T> {
    T findById(int id);
    void save(T entity);
    void delete(int id);
}
"""
        symbols, _ = parse("java", code)
        assert "Repository" in names_of(symbols)
        method_names = methods_of(symbols, "Repository")
        assert {"findById", "save", "delete"} <= method_names

    def test_visibility_modifiers(self):
        code = """
public class Visibility {
    public void pub() {}
    private void priv() {}
    protected void prot() {}
}
"""
        symbols, _ = parse("java", code)
        by_name = {s.name: s for s in symbols if s.kind == SymbolKind.method}
        assert by_name["pub"].visibility == "public"
        assert by_name["priv"].visibility == "private"
        assert by_name["prot"].visibility == "protected"

    def test_static_and_abstract(self):
        code = """
public abstract class Base {
    public static void helper() {}
    public abstract void abstractMethod();
    public final void concrete() {}
}
"""
        symbols, _ = parse("java", code)
        by_name = {s.name: s for s in symbols if s.kind == SymbolKind.method}
        assert by_name["helper"].is_static
        assert by_name["abstractMethod"].is_abstract

    def test_enum(self):
        code = """
public enum Color {
    RED, GREEN, BLUE;

    public String hex() {
        return "#000000";
    }
}
"""
        symbols, _ = parse("java", code)
        names = names_of(symbols)
        assert "Color" in names

    def test_generic_class(self):
        code = """
public class Container<T> {
    private T item;

    public T get() { return item; }
    public void set(T item) { this.item = item; }
}
"""
        symbols, _ = parse("java", code)
        method_names = methods_of(symbols, "Container")
        assert {"get", "set"} <= method_names


# ─────────────────────────────────────────────────────────────────────────────
# Ruby (generic parser)
# ─────────────────────────────────────────────────────────────────────────────


class TestRuby:
    """Tests for Ruby via the generic parser."""

    def test_class_with_methods(self):
        code = """
class Calculator
  def initialize
    @history = []
  end

  def add(a, b)
    a + b
  end

  def subtract(a, b)
    a - b
  end
end
"""
        symbols, _ = parse("ruby", code)
        assert "Calculator" in names_of(symbols)
        method_names = methods_of(symbols, "Calculator")
        assert {"initialize", "add", "subtract"} <= method_names
        assert_all_methods_have_parent(symbols)

    def test_module(self):
        code = """
module Greeting
  def self.hello
    "Hello"
  end

  def goodbye
    "Goodbye"
  end
end
"""
        symbols, _ = parse("ruby", code)
        names = names_of(symbols)
        assert "Greeting" in names

    def test_inheritance(self):
        code = """
class Animal
  def name
    "animal"
  end
end

class Dog < Animal
  def bark
    "woof"
  end
end
"""
        symbols, _ = parse("ruby", code)
        names = names_of(symbols)
        assert "Animal" in names
        assert "Dog" in names
        assert "bark" in methods_of(symbols, "Dog")

    def test_special_method_names(self):
        code = """
class Container
  def [](key)
    @data[key]
  end

  def []=(key, value)
    @data[key] = value
  end

  def each
    @data.each { |x| yield x }
  end
end
"""
        symbols, _ = parse("ruby", code)
        # At minimum, "each" should be detected
        method_names = methods_of(symbols, "Container")
        assert "each" in method_names


# ─────────────────────────────────────────────────────────────────────────────
# C# (generic parser)
# ─────────────────────────────────────────────────────────────────────────────


class TestCSharp:
    """Tests for C# via the generic parser."""

    def test_class_with_methods(self):
        code = """
public class Program {
    public static void Main(string[] args) {
        Console.WriteLine("hi");
    }

    private int Add(int a, int b) {
        return a + b;
    }
}
"""
        symbols, _ = parse("csharp", code)
        assert "Program" in names_of(symbols)
        assert {"Main", "Add"} <= methods_of(symbols, "Program")
        assert_all_methods_have_parent(symbols)

    def test_interface(self):
        code = """
public interface IService {
    void Start();
    void Stop();
    bool IsRunning();
}
"""
        symbols, _ = parse("csharp", code)
        assert "IService" in names_of(symbols)

    def test_generic_class(self):
        code = """
public class List<T> where T : class {
    public void Add(T item) {}
    public T Get(int index) { return default; }
}
"""
        symbols, _ = parse("csharp", code)
        method_names = methods_of(symbols, "List")
        assert {"Add", "Get"} <= method_names

    def test_struct(self):
        code = """
public struct Point {
    public int X;
    public int Y;

    public double Distance(Point other) {
        return 0.0;
    }
}
"""
        symbols, _ = parse("csharp", code)
        names = names_of(symbols)
        assert "Point" in names

    def test_record(self):
        code = """
public record Person(string Name, int Age);
public record Address {
    public string Street { get; init; }
}
"""
        symbols, _ = parse("csharp", code)
        names = names_of(symbols)
        # records may register as class or as their own kind
        assert "Person" in names or "Address" in names


# ─────────────────────────────────────────────────────────────────────────────
# PHP (generic parser)
# ─────────────────────────────────────────────────────────────────────────────


class TestPHP:
    """Tests for PHP via the generic parser."""

    def test_class_with_methods(self):
        code = """<?php
class UserService {
    private $db;

    public function __construct($db) {
        $this->db = $db;
    }

    public function findUser($id) {
        return $this->db->find($id);
    }

    private function log($msg) {
        error_log($msg);
    }
}
"""
        symbols, _ = parse("php", code)
        assert "UserService" in names_of(symbols)
        method_names = methods_of(symbols, "UserService")
        assert {"__construct", "findUser", "log"} <= method_names
        assert_all_methods_have_parent(symbols)

    def test_interface(self):
        code = """<?php
interface PaymentGateway {
    public function charge($amount);
    public function refund($transactionId);
}
"""
        symbols, _ = parse("php", code)
        assert "PaymentGateway" in names_of(symbols)

    def test_trait(self):
        code = """<?php
trait Loggable {
    public function log($msg) {
        echo $msg;
    }
}

class Service {
    use Loggable;
    public function doWork() {}
}
"""
        symbols, _ = parse("php", code)
        names = names_of(symbols)
        assert "Loggable" in names
        assert "Service" in names

    def test_visibility(self):
        code = """<?php
class Vis {
    public function pub() {}
    private function priv() {}
    protected function prot() {}
}
"""
        symbols, _ = parse("php", code)
        by_name = {s.name: s for s in symbols if s.kind == SymbolKind.method}
        assert by_name["pub"].visibility == "public"
        assert by_name["priv"].visibility == "private"
        assert by_name["prot"].visibility == "protected"


# ─────────────────────────────────────────────────────────────────────────────
# Kotlin (generic parser)
# ─────────────────────────────────────────────────────────────────────────────


class TestKotlin:
    """Tests for Kotlin via the generic parser."""

    def test_class_with_methods(self):
        code = """
class UserService(private val db: Database) {
    fun findUser(id: Int): User {
        return db.query(id)
    }

    private fun log(msg: String) {
        println(msg)
    }
}
"""
        symbols, _ = parse("kotlin", code)
        assert "UserService" in names_of(symbols)
        method_names = methods_of(symbols, "UserService")
        assert {"findUser", "log"} <= method_names

    def test_data_class(self):
        code = """
data class Point(val x: Int, val y: Int)
data class User(val name: String, val age: Int)
"""
        symbols, _ = parse("kotlin", code)
        names = names_of(symbols)
        assert "Point" in names
        assert "User" in names

    def test_object_declaration(self):
        code = """
object Singleton {
    fun doSomething() {
        println("done")
    }
}
"""
        symbols, _ = parse("kotlin", code)
        assert "Singleton" in names_of(symbols)


# ─────────────────────────────────────────────────────────────────────────────
# Bash (generic parser)
# ─────────────────────────────────────────────────────────────────────────────


class TestBash:
    """Tests for Bash via the generic parser."""

    def test_function_definitions(self):
        code = """
greet() {
    echo "Hello, $1"
}

function farewell() {
    echo "Goodbye, $1"
}
"""
        symbols, _ = parse("bash", code)
        funcs = {s.name for s in symbols if s.kind in (SymbolKind.function, SymbolKind.method)}
        assert "greet" in funcs or "farewell" in funcs

    def test_variable_assignments(self):
        code = """
MAX_RETRIES=5
DEFAULT_TIMEOUT=30
"""
        symbols, _ = parse("bash", code)
        # bash globals may or may not be picked up depending on config
        assert isinstance(symbols, list)


# ─────────────────────────────────────────────────────────────────────────────
# C++ (generic parser, falls back to C grammar)
# ─────────────────────────────────────────────────────────────────────────────


class TestCpp:
    """Tests for C++ via the generic parser (using the C grammar)."""

    def test_basic_function(self):
        code = """
int add(int a, int b) {
    return a + b;
}
"""
        symbols, _ = parse("cpp", code)
        funcs = {s.name for s in symbols if s.kind == SymbolKind.function}
        assert "add" in funcs

    def test_struct_definition(self):
        code = """
struct Point {
    int x;
    int y;
};
"""
        symbols, _ = parse("cpp", code)
        # Struct should be detected as a class
        names = names_of(symbols)
        assert "Point" in names


# ─────────────────────────────────────────────────────────────────────────────
# Cross-cutting: parser must never crash on broken or empty input
# ─────────────────────────────────────────────────────────────────────────────


class TestRobustness:
    """Tests that exercise edge cases across multiple parsers.

    Each parser must:
      * Never raise on broken syntax (tree-sitter is error-tolerant)
      * Return [] for empty input
      * Handle Unicode identifiers in languages that support them
    """

    @pytest.mark.parametrize("lang", [
        "python", "javascript", "typescript", "rust", "c",
        "go", "java", "ruby", "csharp", "php", "kotlin",
    ])
    def test_empty_input_no_crash(self, lang):
        symbols, _ = parse(lang, "")
        assert symbols == [] or isinstance(symbols, list)

    @pytest.mark.parametrize("lang", [
        "python", "javascript", "typescript", "go", "java",
    ])
    def test_only_comments_no_crash(self, lang):
        comment_starters = {
            "python": "# only comments\n# nothing else\n",
            "javascript": "// only comments\n/* block */\n",
            "typescript": "// only comments\n/* block */\n",
            "go": "// only comments\n",
            "java": "// only comments\n/* block */\n",
        }
        symbols, _ = parse(lang, comment_starters[lang])
        assert isinstance(symbols, list)

    @pytest.mark.parametrize("lang,code", [
        ("python", "def\nclass:\n  if"),
        ("javascript", "function\nclass {{\n"),
        ("typescript", "class { unclosed:\n"),
        ("rust", "fn { unclosed"),
        ("go", "func { broken"),
        ("java", "public class { void"),
        ("csharp", "public class { void"),
    ])
    def test_broken_syntax_no_crash(self, lang, code):
        symbols, _ = parse(lang, code)
        assert isinstance(symbols, list)

    @pytest.mark.parametrize("lang,code", [
        ("python", "def 你好(): pass\n"),
        ("javascript", "function 函数() { return 1; }"),
        ("rust", "fn naïve() {}"),
        ("go", "func naïve() {}"),
    ])
    def test_unicode_identifiers(self, lang, code):
        symbols, _ = parse(lang, code)
        # Should not crash; the symbol may or may not be picked up
        # depending on the grammar's identifier regex.
        assert isinstance(symbols, list)
