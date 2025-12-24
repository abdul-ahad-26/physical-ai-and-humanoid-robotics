import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import { motion } from 'framer-motion';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <motion.h1
          className="hero__title"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          {siteConfig.title}
        </motion.h1>
        <motion.p
          className="hero__subtitle"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          {siteConfig.tagline}
        </motion.p>
        <motion.div
          className={styles.buttons}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <Link
            className={clsx('button button--lg', styles.heroButton)}
            to="/docs/intro">
            Start Learning 📚
          </Link>
        </motion.div>
      </div>
    </header>
  );
}

const moduleCards = [
  {
    title: 'Physical AI',
    description: 'Learn about embodied intelligence and the principles of Physical AI.',
    icon: '🤖',
    image: '/img/module-physical-ai.svg',
    link: '/docs/embodied-intelligence'
  },
  {
    title: 'ROS 2 & Simulation',
    description: 'Master ROS 2 fundamentals and simulation environments like Gazebo and Unity.',
    icon: '⚙️',
    image: '/img/module-ros2.svg',
    link: '/docs/ros2-core'
  },
  {
    title: 'Humanoid Robotics',
    description: 'Explore humanoid kinematics, locomotion, and manipulation techniques.',
    icon: '🦾',
    image: '/img/module-humanoid.svg',
    link: '/docs/humanoid-kinematics'
  }
];

const differentiators = [
  {
    title: 'Physical AI First',
    description: 'Unlike traditional AI textbooks that focus on digital intelligence, this book centers on embodied AI systems that interact with the physical world through sensors, actuators, and real-world feedback loops.',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>
        <path d="M2 12h20"/>
      </svg>
    )
  },
  {
    title: 'Agentic AI Systems',
    description: 'Learn to build autonomous AI agents using the OpenAI Agents SDK. Go beyond simple chatbots to create agents that perceive, reason, plan, and execute actions in robotic systems.',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"/>
        <circle cx="12" cy="12" r="3"/>
      </svg>
    )
  },
  {
    title: 'Spec-Driven Development',
    description: 'Master professional software engineering practices with Spec-Kit Plus. Learn to architect, plan, and implement robotic systems using specifications, ADRs, and structured workflows used in production environments.',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="9" y1="15" x2="15" y2="15"/>
        <line x1="9" y1="12" x2="15" y2="12"/>
        <line x1="9" y1="18" x2="12" y2="18"/>
      </svg>
    )
  },
  {
    title: 'Production-Ready Skills',
    description: 'Build real systems that work beyond the classroom. From ROS 2 deployment to cloud infrastructure, learn the tools and patterns used by robotics engineers in industry, not just academic toy examples.',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    )
  },
  {
    title: 'Co-Learning with AI',
    description: 'Learn how to effectively collaborate with AI tools throughout the development process. This book teaches both robotics and how to leverage AI assistants as thinking partners, not just code generators.',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
        <circle cx="9" cy="7" r="4"/>
        <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
        <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
      </svg>
    )
  },
  {
    title: 'End-to-End Journey',
    description: 'Progress systematically from fundamentals to deployment. Each chapter builds on previous concepts, taking you from basic ROS 2 nodes to fully autonomous humanoid systems with perception, planning, and control.',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <line x1="5" y1="12" x2="19" y2="12"/>
        <polyline points="12 5 19 12 12 19"/>
      </svg>
    )
  }
];

function ModuleCard({ title, description, icon, image, link, index }) {
  return (
    <motion.div
      className="col col--4"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
    >
      <Link to={link} className={styles.moduleCard}>
        <motion.div
          whileHover={{ scale: 1.05, y: -5 }}
          transition={{ duration: 0.3 }}
          className={styles.moduleCardContent}
        >
          {image && (
            <div className={styles.moduleImageWrapper}>
              <img src={image} alt={title} className={styles.moduleImage} />
            </div>
          )}
          <div className={styles.moduleIcon}>{icon}</div>
          <h3 className={styles.moduleTitle}>{title}</h3>
          <p className={styles.moduleDescription}>{description}</p>
        </motion.div>
      </Link>
    </motion.div>
  );
}

function DifferentiatorCard({ title, description, icon, index }) {
  return (
    <motion.div
      className="col col--4"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.5, delay: index * 0.15 }}
    >
      <motion.div
        whileHover={{ y: -8 }}
        transition={{ duration: 0.3 }}
        className={styles.differentiatorCard}
      >
        <div className={styles.differentiatorIcon}>{icon}</div>
        <h3 className={styles.differentiatorTitle}>{title}</h3>
        <p className={styles.differentiatorDescription}>{description}</p>
      </motion.div>
    </motion.div>
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
            <motion.h2
              className={styles.featuresTitle}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              Explore the Curriculum
            </motion.h2>
            <div className="row">
              {moduleCards.map((module, idx) => (
                <ModuleCard key={idx} {...module} index={idx} />
              ))}
            </div>
          </div>
        </section>

        <section className={styles.differentiators}>
          <div className="container">
            <motion.h2
              className={styles.differentiatorsTitle}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              What Makes This Book Different
            </motion.h2>
            <motion.p
              className={styles.differentiatorsSubtitle}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              A new approach to learning Physical AI and Robotics
            </motion.p>
            <div className="row">
              {differentiators.map((diff, idx) => (
                <DifferentiatorCard key={idx} {...diff} index={idx} />
              ))}
            </div>
          </div>
        </section>

        <section className="margin-vert--xl padding-vert--lg">
          <div className="container">
            <div className={clsx("row", styles.centerRow)}>
              <div className="col col--8 col--offset--2">
                <motion.h2
                  className="text--center margin-bottom--lg"
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5 }}
                >
                  About This Textbook
                </motion.h2>
                <motion.p
                  className="text--center"
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  This comprehensive textbook covers the complete spectrum of Physical AI and Humanoid Robotics,
                  from fundamental concepts to advanced implementations. Designed for students, researchers,
                  and practitioners, it provides both theoretical foundations and practical applications.
                </motion.p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}