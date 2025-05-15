import param
import pytest

try:
    import lumen.ai
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from pydantic import BaseModel, Field

from lumen.ai.translate import param_to_pydantic, pydantic_to_param_instance

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.translate import doc_descriptions, function_to_model


def add(a: int, b: int) -> int:
    """
    Adds two integers

    Parameters
    ----------
    a : int
        Integer A
    b : int
        Integer B

    Returns
    -------
    int
        Result of addition
    """
    return a + b


def add_google(a: int, b: int) -> int:
    """
    Adds two integers

    Args:
        a (int): Integer A
        b (int): Integer B

    Returns:
        int: Result of addition
    """
    return a + b


def add_sphinx(a: int, b: int) -> int:
    """
    Adds two integers

    :param a: Integer A
    :type a: int
    :param b: Integer B
    :type b: int
    :return: The sum of the two integers.
    :rtype: int
    """
    return a + b


def test_doc_descriptions_numpy():
    main, descriptions = doc_descriptions(add)
    assert main == "Adds two integers"
    assert descriptions["a"] == "Integer A"
    assert descriptions["b"] == "Integer B"


def test_doc_descriptions_google():
    main, descriptions = doc_descriptions(add_google)
    assert main == "Adds two integers"
    assert descriptions["a"] == "Integer A"
    assert descriptions["b"] == "Integer B"


def test_doc_descriptions_sphinx():
    main, descriptions = doc_descriptions(add_sphinx)
    assert main == "Adds two integers"
    assert descriptions["a"] == "Integer A"
    assert descriptions["b"] == "Integer B"


def test_function_to_model_args():
    model = function_to_model(add)

    fields = model.model_fields
    assert len(fields) == 2
    assert "a" in fields
    assert fields["a"].annotation is int
    assert fields["a"].description == "Integer A"
    assert "b" in fields
    assert fields["b"].annotation is int
    assert fields["b"].description == "Integer B"


def test_param_to_pydantic_conversion():
    """
    Test that param.Parameterized classes are correctly converted to Pydantic models.
    This tests our use of Field() instead of FieldInfo in the translate module.
    """

    # Define direct Pydantic models for comparison
    class PydanticPet(BaseModel):
        name: str = Field(..., description="Name of the pet")
        age: int = Field(..., ge=0, le=20, description="Age of the pet in years")

    class PydanticDog(PydanticPet):
        bark_volume: int = Field(default=5, ge=0, le=10)

    class PydanticCat(PydanticPet):
        meow_pitch: float = Field(default=0.5, gt=0, lt=1)

    class PydanticOwner(BaseModel):
        pet: PydanticDog | PydanticCat

    # Define param.Parameterized classes
    class Pet(param.Parameterized):
        name = param.String(doc="Name of the pet")
        age = param.Integer(default=1, bounds=(0, 20), doc="Age of the pet in years")

    class Dog(Pet):
        bark_volume = param.Integer(default=5, bounds=(0, 10))

    class Cat(Pet):
        meow_pitch = param.Number(default=0.5, bounds=(0, 1))

    class Owner(param.Parameterized):
        pet = param.ClassSelector(class_=(Dog, Cat))

    # Convert param classes to Pydantic models
    created_models = param_to_pydantic(Owner, excluded=["name"])
    ParamOwner = created_models["Owner"]
    ParamDog = created_models["Dog"]
    ParamCat = created_models["Cat"]

    # Get schemas
    param_owner_schema = ParamOwner.model_json_schema()

    # Test pet field in Owner
    assert "pet" in param_owner_schema["properties"]

    # Test field metadata conversion in Dog
    param_dog_schema = ParamDog.model_json_schema()
    assert "bark_volume" in param_dog_schema["properties"]
    bark_volume_prop = param_dog_schema["properties"]["bark_volume"]
    assert bark_volume_prop["default"] == 5
    assert "minimum" in bark_volume_prop
    assert "maximum" in bark_volume_prop
    assert bark_volume_prop["minimum"] == 0
    assert bark_volume_prop["maximum"] == 10

    # Test field metadata conversion in Cat
    param_cat_schema = ParamCat.model_json_schema()
    assert "meow_pitch" in param_cat_schema["properties"]
    meow_pitch_prop = param_cat_schema["properties"]["meow_pitch"]
    assert meow_pitch_prop["default"] == 0.5
    assert "minimum" in meow_pitch_prop
    assert "maximum" in meow_pitch_prop
    assert meow_pitch_prop["minimum"] == 0
    assert meow_pitch_prop["maximum"] == 1

    # Test description preservation
    param_pet_schema = param_dog_schema  # Dog inherits from Pet
    assert "age" in param_pet_schema["properties"]
    age_prop = param_pet_schema["properties"]["age"]
    assert "description" in age_prop
    assert age_prop["description"] == "Age of the pet in years"


def test_field_bounds_conversion():
    """
    Test that parameter bounds are correctly converted to Field constraints.
    """

    class NumberModel(param.Parameterized):
        value = param.Number(default=5, bounds=(0, 10))

    created_models = param_to_pydantic(NumberModel)
    PydanticNumberModel = created_models["NumberModel"]

    # Create an instance to verify constraints work
    model = PydanticNumberModel(value=5)
    assert model.value == 5

    # Check schema
    schema = PydanticNumberModel.model_json_schema()
    assert "value" in schema["properties"]
    value_prop = schema["properties"]["value"]
    assert value_prop["default"] == 5
    assert value_prop["minimum"] == 0
    assert value_prop["maximum"] == 10

    # Verify Field constraints by trying invalid values
    with pytest.raises(Exception):
        PydanticNumberModel(value=-1)  # Below minimum

    with pytest.raises(Exception):
        PydanticNumberModel(value=11)  # Above maximum


def test_alias_handling():
    """
    Test that reserved field names are correctly aliased.
    """

    class ReservedNamesModel(param.Parameterized):
        schema = param.String(default="my_schema")
        copy = param.Boolean(default=True)

    created_models = param_to_pydantic(ReservedNamesModel)
    PydanticReservedModel = created_models["ReservedNamesModel"]

    # First, verify the default values
    model_with_defaults = PydanticReservedModel()
    assert model_with_defaults.schema_ == "my_schema"
    assert model_with_defaults.copy_ == True

    # Create a model with custom values using the field aliases
    # The field name is schema_ but the alias is schema
    model = PydanticReservedModel()
    model.schema_ = "test_schema"
    model.copy_ = False

    # Verify the values were set correctly
    assert model.schema_ == "test_schema"  # Field renamed to schema_
    assert model.copy_ == False  # Field renamed to copy_

    # Check schema to verify aliases
    schema = PydanticReservedModel.model_json_schema()
    assert "schema_" in schema["properties"]
    assert "copy_" in schema["properties"]


def test_various_param_types_conversion():
    """
    Test that various param types are correctly converted to Field objects.
    """

    class AllParamTypes(param.Parameterized):
        # Basic types
        string_param = param.String(default="hello", doc="A string parameter")
        int_param = param.Integer(default=42, bounds=(0, 100), doc="An integer parameter")
        float_param = param.Number(default=3.14, bounds=(0.0, 10.0), doc="A float parameter")
        bool_param = param.Boolean(default=True, doc="A boolean parameter")

        # Collection types
        list_param = param.List(default=[1, 2, 3], doc="A list parameter")
        dict_param = param.Dict(default={"a": 1, "b": 2}, doc="A dictionary parameter")
        tuple_param = param.Tuple(default=(1, "a"), doc="A tuple parameter")

        # Selection types
        selector_param = param.Selector(default="option1", objects=["option1", "option2", "option3"], doc="A selector parameter")

        # Date and time
        date_param = param.Date(default=None, doc="A date parameter")

        # Allow None
        nullable_param = param.String(default=None, allow_None=True, doc="A nullable string parameter")

        # Color param
        color_param = param.Color(default="#FF0000", doc="A color parameter")

    created_models = param_to_pydantic(AllParamTypes)
    PydanticAllParams = created_models["AllParamTypes"]

    # Create an instance with defaults - providing values for the required fields
    instance = PydanticAllParams(date_param=None, nullable_param=None)

    # Check basic types
    assert instance.string_param == "hello"
    assert instance.int_param == 42
    assert instance.float_param == 3.14
    assert instance.bool_param == True

    # Check collection types
    assert instance.list_param == [1, 2, 3]
    assert instance.dict_param == {"a": 1, "b": 2}
    assert instance.tuple_param == (1, "a")

    # Check selector
    assert instance.selector_param == "option1"

    # Check nullable param
    assert instance.nullable_param is None

    # Check schema for bounds and descriptions
    schema = PydanticAllParams.model_json_schema()

    # Check string param
    string_prop = schema["properties"]["string_param"]
    assert string_prop["type"] == "string"
    assert string_prop["default"] == "hello"
    assert string_prop["description"] == "A string parameter"

    # Check int param with bounds
    int_prop = schema["properties"]["int_param"]
    assert int_prop["type"] == "integer"
    assert int_prop["default"] == 42
    assert int_prop["minimum"] == 0
    assert int_prop["maximum"] == 100
    assert int_prop["description"] == "An integer parameter"

    # Check selector param becomes enum
    selector_prop = schema["properties"]["selector_param"]
    assert selector_prop["default"] == "option1"
    assert "enum" in selector_prop
    assert selector_prop["enum"] == ["option1", "option2", "option3"]

    # Check nullable param allows null
    nullable_prop = schema["properties"]["nullable_param"]
    # In Pydantic v2, nullable types might be represented as anyOf/oneOf
    # or with a nullable field instead of type: ["string", "null"]
    if "type" in nullable_prop and isinstance(nullable_prop["type"], list):
        assert "null" in nullable_prop["type"]
    elif "anyOf" in nullable_prop:
        # Check if any schema in anyOf has type null
        null_schema_exists = any(schema.get("type") == "null" for schema in nullable_prop["anyOf"])
        assert null_schema_exists, "No null type found in anyOf schemas"
    elif "nullable" in nullable_prop:
        assert nullable_prop["nullable"] is True


def test_private_attributes():
    """
    Test that attributes with leading underscores are correctly converted to
    Pydantic PrivateAttr objects.
    """

    class ModelWithPrivateAttrs(param.Parameterized):
        _private = param.String(default="private_value")
        public = param.String(default="public_value")

    created_models = param_to_pydantic(ModelWithPrivateAttrs)
    PydanticModel = created_models["ModelWithPrivateAttrs"]

    # Create an instance
    model = PydanticModel()

    # Check that private attributes are directly accessible as attributes
    assert hasattr(model, "_private")
    assert model._private.default == "private_value"
    assert model.public == "public_value"

    # Verify the model schema does NOT include the private attribute
    schema = PydanticModel.model_json_schema()
    assert "_private" not in schema["properties"]
    assert "public" in schema["properties"]


# Round Trip Tests
def test_basic_round_trip():
    """
    Test a basic round trip conversion: param -> pydantic -> param
    """

    # Define a simple Parameterized class
    class SimpleParamClass(param.Parameterized):
        name = param.String(default="test", doc="A test parameter")
        count = param.Integer(default=42, bounds=(0, 100))
        status = param.Boolean(default=True)

    # Create a param instance to start with
    original_param = SimpleParamClass(name="original", count=50, status=False)

    # Convert to pydantic
    created_models = param_to_pydantic(SimpleParamClass)
    PydanticModel = created_models["SimpleParamClass"]

    # Convert back to param
    round_trip_param = pydantic_to_param_instance(PydanticModel(
        name=original_param.name,
        count=original_param.count,
        status=original_param.status
    ))

    # Verify values are preserved
    assert isinstance(round_trip_param, SimpleParamClass)
    assert round_trip_param.name == original_param.name
    assert round_trip_param.count == original_param.count
    assert round_trip_param.status == original_param.status

    # Verify param behaviors are preserved (e.g., bounds checking)
    with pytest.raises(ValueError):
        round_trip_param.count = 101  # Above maximum

    with pytest.raises(ValueError):
        round_trip_param.count = -1  # Below minimum


def test_complex_round_trip():
    """
    Test round trip with complex hierarchy and all parameter types.
    """

    # Define a hierarchy of Parameterized classes
    class Animal(param.Parameterized):
        name = param.String(default="unnamed", doc="Animal name")
        age = param.Integer(default=1, bounds=(0, 30), doc="Age in years")

    class Dog(Animal):
        breed = param.ObjectSelector(default="mixed", objects=["mixed", "labrador", "shepherd"], doc="Dog breed")
        size = param.Selector(default="medium", objects=["small", "medium", "large"], doc="Size category")

    class Person(param.Parameterized):
        name = param.String(default="John Doe", doc="Person name")
        pets = param.List(default=[], item_type=Animal, doc="List of pets")
        favorite_colors = param.List(default=["blue", "green"], doc="Favorite colors")
        preferences = param.Dict(default={"likes_coffee": True, "likes_tea": False}, doc="Preferences")
        birthday = param.Date(default=None, allow_None=True, doc="Birthday")
        profile_color = param.Color(default="#1A85FF", doc="Profile color")

    # Create original param instances
    dog1 = Dog(name="Rex", age=3, breed="labrador", size="large")
    dog2 = Dog(name="Bella", age=2, breed="shepherd", size="medium")
    person = Person(
        name="Alice Smith",
        pets=[dog1, dog2],
        favorite_colors=["purple", "orange"],
        preferences={"likes_coffee": False, "likes_hiking": True},
        birthday=None,
        profile_color="#ff5733",
    )

    # Convert class to pydantic model
    created_models = param_to_pydantic(Person)
    PydanticPerson = created_models["Person"]
    PydanticDog = created_models["Dog"]

    # Convert original dogs to pydantic
    pydantic_dog1 = PydanticDog(name="Rex", age=3, breed="labrador", size="large")

    pydantic_dog2 = PydanticDog(name="Bella", age=2, breed="shepherd", size="medium")

    # Create pydantic person instance manually
    pydantic_person = PydanticPerson(
        name="Alice Smith",
        pets=[pydantic_dog1, pydantic_dog2],
        favorite_colors=["purple", "orange"],
        preferences={"likes_coffee": False, "likes_hiking": True},
        birthday=None,
        profile_color="#ff5733",
    )

    # Convert back to param
    round_trip_person = pydantic_to_param_instance(pydantic_person)

    # Verify all attributes are preserved
    assert isinstance(round_trip_person, Person)
    assert round_trip_person.name == person.name
    assert len(round_trip_person.pets) == 2
    assert round_trip_person.profile_color == "#ff5733"

    # Verify nested objects
    pet1 = round_trip_person.pets[0]
    assert isinstance(pet1, Dog)
    assert pet1.name == "Rex"
    assert pet1.breed == "labrador"
    assert pet1.size == "large"

    # Verify param behavior is preserved
    with pytest.raises(ValueError):
        pet1.age = 31  # Above maximum

    # Verify selector constraint is preserved
    with pytest.raises(ValueError):
        pet1.size = "extra-large"  # Not in allowed values


def test_round_trip_modifications():
    """
    Test round trip with modifications in the pydantic step.
    """

    # Define a param class
    class Config(param.Parameterized):
        name = param.String(default="default-config")
        timeout = param.Integer(default=30, bounds=(1, 300))
        retry_count = param.Integer(default=3, bounds=(0, 10))
        debug_mode = param.Boolean(default=False)
        log_levels = param.ListSelector(default=["error"], objects=["error", "warn", "info", "debug"])

    # Convert to pydantic
    created_models = param_to_pydantic(Config)
    PydanticConfig = created_models["Config"]

    # Create pydantic instance with different values
    pydantic_config = PydanticConfig(name="modified-config", timeout=60, retry_count=5, debug_mode=True, log_levels=["error", "warn", "debug"])

    # Now convert back to param
    round_trip_config = pydantic_to_param_instance(pydantic_config)

    # Verify modified values are preserved
    assert round_trip_config.name == "modified-config"
    assert round_trip_config.timeout == 60
    assert round_trip_config.retry_count == 5
    assert round_trip_config.debug_mode is True
    assert round_trip_config.log_levels == ["error", "warn", "debug"]

    # Verify param behavior is still intact
    with pytest.raises(ValueError):
        round_trip_config.retry_count = 11  # Above maximum

    # Make sure we can modify valid values
    round_trip_config.retry_count = 7
    assert round_trip_config.retry_count == 7


def test_round_trip_with_default_values():
    """
    Test round trip conversion using default values.
    """

    # Define param class with defaults
    class Settings(param.Parameterized):
        name = param.String(default="default-settings")
        max_connections = param.Integer(default=100, bounds=(10, 1000))
        enable_cache = param.Boolean(default=True)
        buffer_size = param.Integer(default=4096)

    # Convert to pydantic model
    created_models = param_to_pydantic(Settings)
    PydanticSettings = created_models["Settings"]

    # Create a pydantic instance with default values
    pydantic_settings = PydanticSettings()

    # Convert back to param
    round_trip_settings = pydantic_to_param_instance(pydantic_settings)

    # Verify default values are preserved
    assert isinstance(round_trip_settings, Settings)
    assert round_trip_settings.name == "default-settings"
    assert round_trip_settings.max_connections == 100
    assert round_trip_settings.enable_cache is True
    assert round_trip_settings.buffer_size == 4096

    # Verify param behavior is preserved
    with pytest.raises(ValueError):
        round_trip_settings.max_connections = 5  # Below minimum
