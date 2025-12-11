"""
Schema Evolution Support for Metadata in RAG + Agentic Backend for AI-Textbook Chatbot.

This module provides functionality for managing schema evolution for metadata:
- Versioning of metadata schemas
- Migration of metadata between versions
- Backward compatibility support
- Schema validation and evolution tracking
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy


class SchemaChangeType(Enum):
    """Types of schema changes"""
    ADD_FIELD = "add_field"
    REMOVE_FIELD = "remove_field"
    MODIFY_FIELD = "modify_field"
    RENAME_FIELD = "rename_field"
    ADD_CONSTRAINT = "add_constraint"
    REMOVE_CONSTRAINT = "remove_constraint"


@dataclass
class SchemaVersion:
    """Represents a version of a schema"""
    version: str
    created_at: datetime
    changes: List[Dict[str, Any]]
    description: str
    compatibility: str  # forward, backward, full, none


@dataclass
class MigrationResult:
    """Result of a schema migration"""
    success: bool
    migrated_data: Dict[str, Any]
    warnings: List[str]
    migration_path: List[str]  # Version path taken


class SchemaEvolutionManager:
    """
    Manages schema evolution for metadata in the system.
    Handles versioning, migration, and validation of metadata schemas.
    """

    def __init__(self):
        self.schema_registry = {}  # Maps schema name to its versions
        self.migration_functions = {}  # Maps version pairs to migration functions
        self.validators = {}  # Maps schema name to validation functions

    def register_schema_version(self, schema_name: str, version: str,
                              schema_definition: Dict[str, Any],
                              description: str = ""):
        """
        Register a new version of a schema.

        Args:
            schema_name: Name of the schema
            version: Version identifier (e.g., "1.0.0")
            schema_definition: JSON Schema definition
            description: Description of the schema version
        """
        if schema_name not in self.schema_registry:
            self.schema_registry[schema_name] = {}

        self.schema_registry[schema_name][version] = SchemaVersion(
            version=version,
            created_at=datetime.utcnow(),
            changes=[],  # Changes would be populated based on diff with previous version
            description=description,
            compatibility="forward"  # Default assumption
        )

    def register_migration_function(self, schema_name: str,
                                  from_version: str,
                                  to_version: str,
                                  migration_func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Register a migration function to transform data from one schema version to another.

        Args:
            schema_name: Name of the schema
            from_version: Source version
            to_version: Target version
            migration_func: Function that takes old data and returns new data
        """
        key = f"{schema_name}:{from_version}:{to_version}"
        self.migration_functions[key] = migration_func

    def register_validator(self, schema_name: str,
                          validator_func: Callable[[Dict[str, Any]], bool]):
        """
        Register a validation function for a schema.

        Args:
            schema_name: Name of the schema
            validator_func: Function that validates the data
        """
        self.validators[schema_name] = validator_func

    def get_current_schema_version(self, schema_name: str) -> Optional[str]:
        """
        Get the current (latest) version of a schema.

        Args:
            schema_name: Name of the schema

        Returns:
            Latest version string or None if schema doesn't exist
        """
        if schema_name not in self.schema_registry:
            return None

        versions = list(self.schema_registry[schema_name].keys())
        # Sort versions assuming semantic versioning (this is simplified)
        versions.sort(key=lambda x: [int(part) for part in x.split('.')])
        return versions[-1] if versions else None

    def migrate_data(self, schema_name: str, data: Dict[str, Any],
                    target_version: Optional[str] = None) -> MigrationResult:
        """
        Migrate data to the target schema version.

        Args:
            schema_name: Name of the schema
            data: Data to migrate (with optional _schema_version field)
            target_version: Target version (defaults to current version)

        Returns:
            MigrationResult with success status and migrated data
        """
        warnings = []

        # Determine source version from data or assume latest
        source_version = data.get("_schema_version", "1.0.0")

        if target_version is None:
            target_version = self.get_current_schema_version(schema_name)

        if not target_version:
            return MigrationResult(
                success=False,
                migrated_data=data,
                warnings=["No target version specified and no current version found"],
                migration_path=[]
            )

        # Check if migration is needed
        if source_version == target_version:
            return MigrationResult(
                success=True,
                migrated_data=data,
                warnings=[],
                migration_path=[source_version]
            )

        # Find migration path (simplified - assumes direct migration exists)
        migration_path = [source_version]
        current_version = source_version
        migrated_data = copy.deepcopy(data)

        # Remove schema version from data before migration
        if "_schema_version" in migrated_data:
            del migrated_data["_schema_version"]

        # Perform migration
        migration_key = f"{schema_name}:{current_version}:{target_version}"
        if migration_key in self.migration_functions:
            try:
                migrated_data = self.migration_functions[migration_key](migrated_data)
                migration_path.append(target_version)

                # Add the new schema version to the migrated data
                migrated_data["_schema_version"] = target_version

                return MigrationResult(
                    success=True,
                    migrated_data=migrated_data,
                    warnings=warnings,
                    migration_path=migration_path
                )
            except Exception as e:
                warnings.append(f"Migration failed: {str(e)}")
                return MigrationResult(
                    success=False,
                    migrated_data=data,  # Return original data on failure
                    warnings=warnings,
                    migration_path=migration_path
                )
        else:
            # No direct migration path found
            warnings.append(f"No migration path from {source_version} to {target_version}")
            return MigrationResult(
                success=False,
                migrated_data=data,
                warnings=warnings,
                migration_path=migration_path
            )

    def validate_data(self, schema_name: str, data: Dict[str, Any]) -> bool:
        """
        Validate data against the current schema.

        Args:
            schema_name: Name of the schema
            data: Data to validate

        Returns:
            True if valid, False otherwise
        """
        if schema_name in self.validators:
            return self.validators[schema_name](data)
        else:
            # If no validator is registered, assume data is valid
            return True

    def get_schema_compatibility(self, schema_name: str, version: str) -> str:
        """
        Get the compatibility level of a schema version.

        Args:
            schema_name: Name of the schema
            version: Schema version

        Returns:
            Compatibility level as string
        """
        if (schema_name in self.schema_registry and
            version in self.schema_registry[schema_name]):
            return self.schema_registry[schema_name][version].compatibility
        return "unknown"

    def create_backward_compatible_wrapper(self, schema_name: str):
        """
        Create a wrapper function that handles schema evolution transparently.

        Args:
            schema_name: Name of the schema

        Returns:
            A function that accepts data of any version and returns current version
        """
        def wrapper(data: Dict[str, Any]) -> Dict[str, Any]:
            # Migrate data to current version
            result = self.migrate_data(schema_name, data)
            if result.success:
                return result.migrated_data
            else:
                # If migration fails, return original data with warning
                print(f"Warning: Could not migrate data for {schema_name}: {result.warnings}")
                return data

        return wrapper


class MetadataSchemaEvolution:
    """
    Specific implementation for evolving metadata schemas in the RAG system.
    """

    def __init__(self, evolution_manager: SchemaEvolutionManager):
        self.manager = evolution_manager
        self._setup_default_schemas()

    def _setup_default_schemas(self):
        """Setup default schemas for the RAG system"""
        # Define version 1.0.0 schema for textbook content metadata
        v1_0_0_schema = {
            "type": "object",
            "properties": {
                "source_file": {"type": "string"},
                "section": {"type": "string"},
                "document_type": {"type": "string", "enum": ["markdown", "html"]},
                "page_number": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["source_file", "document_type"]
        }

        # Define version 2.0.0 schema with additional fields
        v2_0_0_schema = {
            "type": "object",
            "properties": {
                "source_file": {"type": "string"},
                "section": {"type": "string"},
                "document_type": {"type": "string", "enum": ["markdown", "html", "pdf"]},
                "page_number": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "subject_area": {"type": "string"},  # New field
                "grade_level": {"type": "string"},   # New field
                "learning_objectives": {"type": "array", "items": {"type": "string"}},  # New field
                "difficulty_level": {"type": "integer", "minimum": 1, "maximum": 5}  # New field
            },
            "required": ["source_file", "document_type", "subject_area"]
        }

        # Register schemas
        self.manager.register_schema_version(
            "textbook_content_metadata",
            "1.0.0",
            v1_0_0_schema,
            "Initial version of textbook content metadata schema"
        )

        self.manager.register_schema_version(
            "textbook_content_metadata",
            "2.0.0",
            v2_0_0_schema,
            "Enhanced version with subject area, grade level, and learning objectives"
        )

        # Register migration function from 1.0.0 to 2.0.0
        def migrate_v1_to_v2(data: Dict[str, Any]) -> Dict[str, Any]:
            # Copy the original data
            new_data = copy.deepcopy(data)

            # Add default values for new fields
            new_data["subject_area"] = new_data.get("subject_area", "general")
            new_data["grade_level"] = new_data.get("grade_level", "university")
            new_data["learning_objectives"] = new_data.get("learning_objectives", [])
            new_data["difficulty_level"] = new_data.get("difficulty_level", 3)

            # If document_type was restricted to markdown/html in v1,
            # ensure it's still valid for v2 (it is, so no change needed)

            return new_data

        self.manager.register_migration_function(
            "textbook_content_metadata",
            "1.0.0",
            "2.0.0",
            migrate_v1_to_v2
        )

        # Register validation function
        def validate_textbook_metadata(data: Dict[str, Any]) -> bool:
            # Basic validation - in a real system, you'd use a proper JSON Schema validator
            required_fields = ["source_file", "document_type"]
            for field in required_fields:
                if field not in data:
                    return False

            # Validate document_type if present
            doc_type = data.get("document_type")
            if doc_type and doc_type not in ["markdown", "html", "pdf"]:
                return False

            return True

        self.manager.register_validator("textbook_content_metadata", validate_textbook_metadata)

    def evolve_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolve metadata to the current schema version.

        Args:
            metadata: Metadata to evolve

        Returns:
            Evolved metadata in current schema version
        """
        result = self.manager.migrate_data("textbook_content_metadata", metadata)
        if result.success:
            return result.migrated_data
        else:
            # If migration fails, return original metadata with schema version added
            metadata["_schema_version"] = self.manager.get_current_schema_version("textbook_content_metadata") or "1.0.0"
            print(f"Warning: Schema evolution failed: {result.warnings}")
            return metadata

    def register_new_schema_version(self, version: str, changes: List[Dict[str, Any]],
                                   description: str = ""):
        """
        Register a new version of the textbook content metadata schema.

        Args:
            version: New version string (e.g., "3.0.0")
            changes: List of changes made in this version
            description: Description of the new version
        """
        # This would typically involve creating a new schema definition
        # and registering the appropriate migration functions
        current_version = self.manager.get_current_schema_version("textbook_content_metadata")
        print(f"Registering new schema version {version} (from {current_version}). Changes: {changes}")

        # In a real implementation, you would:
        # 1. Define the new schema
        # 2. Register migration functions from all previous versions to this version
        # 3. Register a validator for the new schema
        pass

    def get_evolution_path(self, from_version: str, to_version: str) -> List[str]:
        """
        Get the evolution path between two versions.

        Args:
            from_version: Starting version
            to_version: Target version

        Returns:
            List of versions representing the evolution path
        """
        # Simplified implementation - in reality, you'd have a more sophisticated path-finding algorithm
        return [from_version, to_version]


class SchemaEvolutionMiddleware:
    """
    Middleware to automatically handle schema evolution for incoming data.
    """

    def __init__(self, evolution_manager: SchemaEvolutionManager):
        self.evolution_manager = evolution_manager

    def process_incoming_metadata(self, metadata: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """
        Process incoming metadata through schema evolution.

        Args:
            metadata: Incoming metadata
            schema_name: Name of the schema to use

        Returns:
            Processed metadata in current schema version
        """
        # Migrate to current version
        result = self.evolution_manager.migrate_data(schema_name, metadata)

        if not result.success:
            print(f"Warning: Schema migration failed for {schema_name}: {result.warnings}")
            # Add current schema version to data
            current_version = self.evolution_manager.get_current_schema_version(schema_name)
            if current_version:
                metadata["_schema_version"] = current_version
            return metadata

        return result.migrated_data

    def validate_and_evolve(self, data: Dict[str, Any], schema_name: str) -> tuple[bool, Dict[str, Any]]:
        """
        Validate and evolve data to current schema.

        Args:
            data: Data to validate and evolve
            schema_name: Name of the schema

        Returns:
            Tuple of (is_valid, evolved_data)
        """
        # First evolve the schema
        evolved_data = self.process_incoming_metadata(data, schema_name)

        # Then validate
        is_valid = self.evolution_manager.validate_data(schema_name, evolved_data)

        return is_valid, evolved_data


# Global schema evolution manager instance
schema_evolution_manager = SchemaEvolutionManager()
metadata_evolution = MetadataSchemaEvolution(schema_evolution_manager)
evolution_middleware = SchemaEvolutionMiddleware(schema_evolution_manager)


def get_schema_evolution_manager() -> SchemaEvolutionManager:
    """Get the global schema evolution manager instance"""
    return schema_evolution_manager


def evolve_textbook_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Evolve textbook metadata to current schema version"""
    return metadata_evolution.evolve_metadata(metadata)


def process_metadata_through_evolution(metadata: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
    """Process metadata through evolution and validation"""
    return evolution_middleware.validate_and_evolve(
        metadata,
        "textbook_content_metadata"
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing Schema Evolution System...")

    # Test with old version metadata (v1.0.0)
    old_metadata = {
        "source_file": "intro_ml_chapter.md",
        "section": "Chapter 1",
        "document_type": "markdown",
        "page_number": 15,
        "tags": ["machine-learning", "introduction"]
    }

    print(f"Original metadata: {old_metadata}")

    # Evolve the metadata
    evolved_metadata = evolve_textbook_metadata(old_metadata)
    print(f"Evolved metadata: {evolved_metadata}")

    # Test validation
    is_valid, validated_data = process_metadata_through_evolution(evolved_metadata)
    print(f"Is valid: {is_valid}, Validated data: {validated_data}")

    # Test with current version metadata
    current_metadata = {
        "source_file": "advanced_ml_chapter.md",
        "section": "Chapter 5",
        "document_type": "markdown",
        "page_number": 125,
        "tags": ["deep-learning", "neural-networks"],
        "subject_area": "Computer Science",
        "grade_level": "graduate",
        "learning_objectives": ["Understand neural networks", "Implement backpropagation"],
        "difficulty_level": 4
    }

    print(f"\nCurrent metadata: {current_metadata}")
    evolved_current = evolve_textbook_metadata(current_metadata)
    print(f"After evolution (should be unchanged): {evolved_current}")

    print("\nSchema Evolution System test completed!")