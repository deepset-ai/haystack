// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from 'react';
import Image from '@theme/IdealImage';
import styles from './styles.module.css';

interface ClickableImageProps {
  src: string; // must start with /img/
  alt?: string;
  className?: string;
  size?: 'standard' | 'large';
}

const images = require.context('@site/static/img', false, /\.(png|jpe?g)$/);

export default function ClickableImage({
  src,
  alt = '',
  className,
  size = 'standard',
}: ClickableImageProps) {
  const [isZoomed, setZoomed] = useState(false);
  const img = images('./' + src.replace(/^\/img\//, ''));
  const sizeClass = size === 'large' ? styles.imgLarge : styles.imgStandard;

  return (
    <>
      <div
        className={`${styles.imageWrapper} ${className || ''}`}
        onClick={() => setZoomed(!isZoomed)}
        role="button"
        aria-pressed={isZoomed}
        title="Click to enlarge"
      >
        <Image img={img} alt={alt} className={`${styles.zoomable} ${sizeClass}`} />
      </div>

      {isZoomed && (
        <div className={styles.overlay} onClick={() => setZoomed(false)}>
          <img src={img.default} alt={alt} className={styles.zoomedImage} />
        </div>
      )}
    </>
  );
}
