/**
 * DocItem Theme Override
 *
 * Simple wrapper that passes through to the original DocItem.
 * The actual personalization integration happens in DocItem/Content/index.tsx
 */

import React from 'react';
import DocItem from '@theme-original/DocItem';
import type DocItemType from '@theme/DocItem';
import type { WrapperProps } from '@docusaurus/types';

type Props = WrapperProps<typeof DocItemType>;

export default function DocItemWrapper(props: Props): JSX.Element {
  return <DocItem {...props} />;
}
