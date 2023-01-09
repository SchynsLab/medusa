import toml
import click

@click.command()
@click.argument('device_type', type=click.Choice(['cpu', 'gpu']))
@click.option('-d', '--device', default='cu116', help='CUDA version (if "gpu")')
def main(device_type, device):
    """Fills ./pyproject_template.toml with the relevant information and
    creates the corresponding ./pyproject.toml file which can be used by
    poetry."""

    if device_type == 'cpu':
        package_name = 'medusa'
    else:
        package_name = 'medusa-gpu'

    cfg = toml.load('./pyproject_template.toml')
    cfg['tool']['poetry']['name'] = package_name
    n = len(cfg['tool']['poetry']['dependencies']['torch'])
    for i in range(n):
        cfg['tool']['poetry']['dependencies']['torch'][i]['url'] = cfg['tool']['poetry']['dependencies']['torch'][i]['url'].format(device=device)

    n = len(cfg['tool']['poetry']['dependencies']['torchvision'])
    for i in range(n):
        cfg['tool']['poetry']['dependencies']['torchvision'][i]['url'] = cfg['tool']['poetry']['dependencies']['torchvision'][i]['url'].format(device=device)

    if device_type == 'cpu':
        # At the moment, pytorch3d does not provide CPU wheels
        del cfg['tool']['poetry']['dependencies']['pytorch3d']
    else:
        cfg['tool']['poetry']['dependencies']['pytorch3d'][0]['url'] = cfg['tool']['poetry']['dependencies']['pytorch3d'][0]['url'].format(device=device)

    with open('./pyproject.toml', 'w') as f_out:
        toml.dump(f=f_out, o=cfg)


if __name__ == '__main__':
    main()
