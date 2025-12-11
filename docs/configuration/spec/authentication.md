# Securing dashboards with authentication

Restrict dashboard access to authorized users.

!!! note "Documentation in progress"
    Detailed authentication documentation is coming soon. For now, see the Panel authentication guide.

## Quick start

Add basic authentication to your dashboard:

```yaml
config:
  auth:
    type: basic
    users:
      - alice
      - bob
```

## Authentication types

| Type | Description | Use case |
|------|-------------|----------|
| `basic` | Username list | Simple user restrictions |
| `oauth` | OAuth providers | GitHub, Google, Azure AD |
| `password` | Username/password | Custom authentication |

## GitHub OAuth example

Restrict to specific GitHub users:

```yaml
config:
  auth:
    type: oauth
    oauth_provider: github
    oauth_key: {{ env("GITHUB_CLIENT_ID") }}
    oauth_secret: {{ env("GITHUB_CLIENT_SECRET") }}
    authorized_users:
      - github_username1
      - github_username2
```

## Resources

For complete authentication documentation, see:

- [Panel Authentication Guide](https://panel.holoviz.org/user_guide/Authentication.html)
- [OAuth Setup Guide](https://panel.holoviz.org/user_guide/Authentication.html#oauth)

## Common patterns

### Development vs production

```yaml
config:
  auth:
    type: {{ env("AUTH_TYPE", "basic") }}
    users: {{ env("AUTH_USERS", "['dev_user']") }}
```

Set different auth for each environment:

```bash
# Development
export AUTH_TYPE=basic
export AUTH_USERS="['alice', 'bob']"

# Production
export AUTH_TYPE=oauth
export OAUTH_PROVIDER=github
```

### Per-layout restrictions

```yaml
layouts:
  - title: Public Dashboard
    # No auth required
    views:
      - type: table

  - title: Admin Dashboard
    auth:
      users: [admin]
    views:
      - type: table
```

## Next steps

- **[Panel Authentication Guide](https://panel.holoviz.org/user_guide/Authentication.html)** - Complete authentication documentation
- **[Deployment guide](deployment.md)** - Deploy secured dashboards
- **[Variables guide](variables.md)** - Use environment variables for secrets
