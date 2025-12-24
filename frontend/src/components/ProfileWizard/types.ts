/**
 * Types for the Profile Wizard component
 *
 * Used in the 3-step profile completion wizard for new users.
 */

export type SoftwareLevel = 'beginner' | 'intermediate' | 'advanced';
export type HardwareLevel = 'none' | 'basic' | 'intermediate' | 'advanced';

export interface SoftwareBackground {
  level: SoftwareLevel;
  languages: string[];
  frameworks: string[];
}

export interface HardwareBackground {
  level: HardwareLevel;
  domains: string[];
}

export interface UserProfile {
  id: string;
  email: string;
  display_name: string | null;
  auth_provider: 'email' | 'google' | 'github';
  software_background: SoftwareBackground | null;
  hardware_background: HardwareBackground | null;
  profile_completed: boolean;
}

export interface ProfileUpdateRequest {
  display_name?: string;
  software_background?: SoftwareBackground;
  hardware_background?: HardwareBackground;
}

export interface ProfileUpdateResponse {
  success: boolean;
  user: UserProfile;
}

export interface WizardState {
  step: 1 | 2 | 3;
  displayName: string;
  softwareBackground: SoftwareBackground;
  hardwareBackground: HardwareBackground;
  isOpen: boolean;
  isSkipped: boolean;
  isLoading: boolean;
  error: string | null;
}

// Available options for dropdowns
export const SOFTWARE_LEVELS: { value: SoftwareLevel; label: string }[] = [
  { value: 'beginner', label: 'Beginner - New to programming' },
  { value: 'intermediate', label: 'Intermediate - Some experience' },
  { value: 'advanced', label: 'Advanced - Professional developer' },
];

export const HARDWARE_LEVELS: { value: HardwareLevel; label: string }[] = [
  { value: 'none', label: 'None - No hardware experience' },
  { value: 'basic', label: 'Basic - Played with Arduino/Raspberry Pi' },
  { value: 'intermediate', label: 'Intermediate - Built hardware projects' },
  { value: 'advanced', label: 'Advanced - Professional hardware engineer' },
];

export const SOFTWARE_LANGUAGES: string[] = [
  'Python',
  'JavaScript',
  'TypeScript',
  'C++',
  'Java',
  'Go',
  'Rust',
  'C',
  'MATLAB',
  'Julia',
];

export const SOFTWARE_FRAMEWORKS: string[] = [
  'TensorFlow',
  'PyTorch',
  'React',
  'FastAPI',
  'ROS',
  'ROS2',
  'Unity',
  'Gazebo',
  'Isaac Sim',
  'OpenCV',
];

export const HARDWARE_DOMAINS: { value: string; label: string }[] = [
  { value: 'basic_electronics', label: 'Basic Electronics' },
  { value: 'robotics_kits', label: 'Robotics Kits (Arduino, Raspberry Pi)' },
  { value: 'gpus_accelerators', label: 'GPUs & Accelerators' },
  { value: 'jetson_edge', label: 'NVIDIA Jetson / Edge Devices' },
  { value: 'embedded_systems', label: 'Embedded Systems' },
  { value: 'sensors_actuators', label: 'Sensors & Actuators' },
  { value: '3d_printing', label: '3D Printing' },
  { value: 'pcb_design', label: 'PCB Design' },
];
