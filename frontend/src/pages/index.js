import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Read the Book - 5min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Home`}
      description="Physical AI & Humanoid Robotics — AI-Native Technical Textbook">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className="col col--4">
                <div className="text--center padding-horiz--md">
                  <h2>Physical AI</h2>
                  <p>Learn about embodied intelligence and the principles of Physical AI.</p>
                </div>
              </div>
              <div className="col col--4">
                <div className="text--center padding-horiz--md">
                  <h2>ROS 2 & Simulation</h2>
                  <p>Master ROS 2 fundamentals and simulation environments like Gazebo and Unity.</p>
                </div>
              </div>
              <div className="col col--4">
                <div className="text--center padding-horiz--md">
                  <h2>Humanoid Robotics</h2>
                  <p>Explore humanoid kinematics, locomotion, and manipulation techniques.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="margin-vert--xl padding-vert--lg">
          <div className="container">
            <div className={clsx("row", styles.centerRow)}>
              <div className="col col--8 col--offset--2">
                <h2 className="text--center margin-bottom--lg">
                  About This Textbook
                </h2>
                <p className="text--center">
                  This comprehensive textbook covers the complete spectrum of Physical AI and Humanoid Robotics,
                  from fundamental concepts to advanced implementations. Designed for students, researchers,
                  and practitioners, it provides both theoretical foundations and practical applications.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}