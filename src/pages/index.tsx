import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        {/* <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Docusaurus Tutorial - 5min ⏱️
          </Link>
        </div> */}
      </div>
    </header>
  );
}

const UnderConstruction = () => {
  const styles = {
    underConstruction: {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '25vh',
      backgroundImage: 'url(https://example.com/path-to-your-image.jpg)', // Replace with your image URL
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      color: 'white',
      textAlign: 'center',
    },
    content: {
      background: 'rgba(0, 0, 0, 0.7)',
      padding: '20px',
      borderRadius: '10px',
    },
    heading: {
      fontSize: '3em',
      marginBottom: '0.5em',
    },
    paragraph: {
      fontSize: '1.2em',
    },
  };

  return (
    <div style={styles.underConstruction}>
      <div style={styles.content}>
        <h1 style={styles.heading}>Site Under Construction</h1>
        <p style={styles.paragraph}>You may see gaps in between, I'll try to fill it with contents.</p>
      </div>
    </div>
  );
};

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Home" /*{siteConfig.title}*/
      description="My PG Work docs">
      <HomepageHeader />
      <UnderConstruction />
      {/* <main>
        <HomepageFeatures />
      </main> */}
    </Layout>
  );
}
