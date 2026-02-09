import { motion } from 'framer-motion'
import PropTypes from 'prop-types'


export default function Card({
    children,
    className = '',
    hover = true,
    ...props
}) {
    return (
        <motion.div
            whileHover={hover ? { y: -2 } : {}}
            className={`
        bg-midnight-800/50 backdrop-blur-sm
        border border-midnight-700 rounded-xl
        ${hover ? 'card-hover' : ''}
        ${className}
      `}
            {...props}
        >
            {children}
        </motion.div>
    )
}

Card.propTypes = {
    children: PropTypes.node.isRequired,
    className: PropTypes.string,
    hover: PropTypes.bool,
}
