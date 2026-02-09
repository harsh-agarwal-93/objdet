import PropTypes from 'prop-types'

export default function Select({
    label,
    options = [],
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
            <select
                className={`
          w-full px-3 py-2 rounded-lg
          bg-midnight-800 border border-midnight-600
          text-white cursor-pointer
          focus:border-neon-teal/50 transition-colors
          ${className}
        `}
                {...props}
            >
                {options.map((option) => (
                    <option key={option.value} value={option.value}>
                        {option.label}
                    </option>
                ))}
            </select>
        </div>
    )
}

Select.propTypes = {
    label: PropTypes.string,
    options: PropTypes.arrayOf(
        PropTypes.shape({
            value: PropTypes.string.isRequired,
            label: PropTypes.string.isRequired,
        })
    ),
    className: PropTypes.string,
}
