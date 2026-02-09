import PropTypes from 'prop-types'

export default function Input({

    label,
    className = '',
    ...props
}) {
    return (
        <div className="space-y-1.5">
            {label && (
                <label className="block text-sm font-medium text-gray-400">
                    {label}
                </label>
            )}
            <input
                className={`
          w-full px-3 py-2 rounded-lg
          bg-midnight-800 border border-midnight-600
          text-white placeholder-gray-500
          focus:border-neon-teal/50 transition-colors
          ${className}
        `}
                {...props}
            />
        </div>
    )
}

Input.propTypes = {
    label: PropTypes.string,
    className: PropTypes.string,
}
