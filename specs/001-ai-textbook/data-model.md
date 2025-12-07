# Data Model: Physical AI & Humanoid Robotics Textbook

## Entity: Textbook Chapter
- **Name**: String (required) - The chapter name/title
- **Description**: String (required) - Brief description of the chapter content
- **Learning Objectives**: Array of strings (required) - List of learning objectives for the chapter
- **Content**: String (required) - Main content in Markdown format
- **Code Examples**: Array of objects - Code snippets with language and content
- **Lab Exercises**: Array of objects - Hands-on exercises with instructions
- **Troubleshooting Tips**: Array of strings - Common issues and solutions
- **Related Chapters**: Array of strings - Cross-references to related chapters
- **Prerequisites**: Array of strings - Required knowledge or chapters to complete first

## Entity: Code Example
- **Language**: String (required) - Programming language (python, ros2, isaac, etc.)
- **Code**: String (required) - The actual code snippet
- **Description**: String - Explanation of what the code does
- **Chapter Reference**: String (required) - Which chapter this example belongs to
- **Tags**: Array of strings - Keywords for categorization and search

## Entity: Lab Exercise
- **Title**: String (required) - The exercise title
- **Description**: String (required) - Detailed instructions for the exercise
- **Objectives**: Array of strings - What the student should learn from this exercise
- **Prerequisites**: Array of strings - What is needed to complete the exercise
- **Steps**: Array of objects - Sequential steps to complete the exercise
- **Expected Results**: String - What the student should expect to see
- **Chapter Reference**: String (required) - Which chapter this exercise belongs to

## Entity: Asset
- **Type**: String (required) - Type of asset (image, diagram, figure, video)
- **Path**: String (required) - File path to the asset
- **Alt Text**: String - Accessibility text for the asset
- **Caption**: String - Descriptive caption
- **Chapter Reference**: String (required) - Which chapter this asset belongs to
- **Usage Context**: String - Where and how the asset is used

## Entity: Simulation Environment
- **Name**: String (required) - Name of the simulation environment (Gazebo, Unity, Isaac Sim)
- **Description**: String - Overview of the simulation environment
- **Setup Instructions**: String - How to set up the environment
- **Dependencies**: Array of strings - Required software/packages
- **Example Scenarios**: Array of strings - Use cases and examples
- **Chapter Reference**: String (required) - Which chapter this environment is used in

## Validation Rules
- Each chapter must have at least one learning objective
- Each chapter must have content (minimum 500 words)
- Code examples must be syntactically correct for their language
- Lab exercises must have clear, sequential steps
- All cross-references must point to existing chapters
- Asset paths must be valid file locations

## State Transitions
- Draft → Review (when initial content is completed)
- Review → Approved (when content passes quality checks)
- Approved → Published (when content is added to textbook)
- Published → Updated (when content needs revision)